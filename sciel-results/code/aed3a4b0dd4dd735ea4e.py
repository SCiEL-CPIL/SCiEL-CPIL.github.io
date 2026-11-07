import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj
        
        # State
        evals_used = 0
        archive_X = []
        archive_F = []
        
        def evaluate(x):
            nonlocal evals_used
            if evals_used >= budget or time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=np.float64)
            evals_used += 1
            return f
            
        def dominates(f1, f2):
            # f1 dominates f2 if f1 <= f2 and f1 != f2
            return np.all(f1 <= f2) and np.any(f1 < f2)
            
        def update_archive(x, f):
            if f is None:
                return
            # Check if f is dominated by any in archive
            # If yes, discard
            for ax, af in zip(archive_X, archive_F):
                if dominates(af, f):
                    return
                    
            # Remove points from archive dominated by f
            to_remove = []
            for i, (ax, af) in enumerate(zip(archive_X, archive_F)):
                if dominates(f, af):
                    to_remove.append(i)
                    
            for idx in sorted(to_remove, reverse=True):
                archive_X.pop(idx)
                archive_F.pop(idx)
                
            archive_X.append(x)
            archive_F.append(f)

        # 1. Initialization with Latin Hypercube for better spread
        n_init = 40
        # LHS: divide each dimension into n_init equal intervals, pick one random point per interval
        lhs_points = np.zeros((n_init, dim))
        for d in range(dim):
            intervals = np.linspace(0, 1, n_init + 1)
            for i in range(n_init):
                lhs_points[i, d] = intervals[i] + np.random.random() * (intervals[i+1] - intervals[i])
        
        X_init = lo + span * lhs_points
        for x in X_init:
            f = evaluate(x)
            if f is not None:
                update_archive(x, f)
        
        # 2. Global Phase: Differential Evolution (DE)
        # Use the non-dominated archive to guide a DE-like search.
        # This maintains diversity by re-sampling the archive as the population.
        
        pop_size = 20
        # Initialize population from archive (or LHS if archive is small)
        if len(archive_X) >= pop_size:
            indices = np.random.choice(len(archive_X), pop_size, replace=False)
            Pop_X = [np.array(archive_X[i]) for i in indices]
        else:
            Pop_X = [np.array(x) for x in X_init]
            
        F_diff = 0.8
        CR = 0.9
        
        # Reserve 60 evals for local polish
        while evals_used < budget - 60 and time.time() < deadline:
            # Generate children for each parent in current Pop
            for i in range(len(Pop_X)):
                if evals_used >= budget - 60 or time.time() > deadline:
                    break
                
                # Select 3 distinct random indices from Pop
                if len(Pop_X) >= 3:
                    r_indices = np.random.choice(len(Pop_X), 3, replace=False)
                    r1, r2, r3 = r_indices
                else:
                    # Fallback to random mutation if pop is too small
                    r1 = np.random.randint(len(Pop_X))
                    r2 = np.random.randint(len(Pop_X))
                    r3 = np.random.randint(len(Pop_X))
                
                # Differential mutation
                v = Pop_X[r1] + F_diff * (Pop_X[r2] - Pop_X[r3])
                
                # Crossover (DE/1/bin)
                j_rand = np.random.randint(dim)
                trial = np.zeros(dim)
                for j in range(dim):
                    if np.random.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = Pop_X[i][j]
                        
                trial = np.clip(trial, lo, hi)
                f_trial = evaluate(trial)
                
                if f_trial is not None:
                    update_archive(trial, f_trial)
            
            # Refresh Population from Archive for next generation
            # This ensures the population always consists of current best non-dominated points
            if len(archive_X) >= pop_size:
                indices = np.random.choice(len(archive_X), pop_size, replace=False)
                Pop_X = [np.array(archive_X[i]) for i in indices]
            else:
                Pop_X = [np.array(x) for x in archive_X]
                
        # 3. Local Polish with Tchebycheff decomposition
        # Use 20 diverse weight vectors to systematically map the entire Pareto front
        if len(archive_X) > 0:
            F_arr = np.array(archive_F)
            
            # Nadir point estimate: max of current archive
            z_max = np.max(F_arr, axis=0) if len(F_arr) > 0 else np.ones(n_obj)
            z_max = z_max + 1e-6 * np.ones(n_obj)
            
            # Define 20 diverse weight vectors for Tchebycheff decomposition
            # Evenly spaced in [0.01, 0.99] to cover the front well
            n_weights = 20
            weight_vectors = [np.array([w, 1-w]) for w in np.linspace(0.01, 0.99, n_weights)]
            
            # For each weight vector, find a good starting point from the archive
            # and run a local optimization
            for w_idx, w in enumerate(weight_vectors):
                if evals_used >= budget or time.time() > deadline:
                    break
                
                # Find the archive point that is best for this weight vector
                # This gives us a good starting point
                best_score = np.inf
                best_x0 = np.array(archive_X[0])
                for ax, af in zip(archive_X, archive_F):
                    score = np.max(w * (af - z_max))
                    if score < best_score:
                        best_score = score
                        best_x0 = np.array(ax)
                
                # Track evaluations in the local polish to stay within budget
                local_evals = 0
                max_local_evals = 8
                
                def obj(x, w=w, z_max=z_max):
                    nonlocal local_evals
                    if local_evals >= max_local_evals or evals_used >= budget or time.time() > deadline:
                        return np.inf
                    x_clipped = np.clip(x, lo, hi)
                    f = np.array(problem(x_clipped), dtype=np.float64)
                    local_evals += 1
                    evals_used += 1
                    # Tchebycheff: max(w * (f - z))
                    return np.max(w * (f - z_max))
                    
                try:
                    res = minimize(obj, best_x0, method='Nelder-Mead', 
                                   options={'maxiter': 50, 'maxfev': max_local_evals, 'xatol': 1e-10, 'fatol': 1e-10})
                    # Evaluate res.x explicitly to get its true F and add to archive
                    if evals_used < budget:
                        x_best = np.clip(res.x, lo, hi)
                        f_best = evaluate(x_best)
                        if f_best is not None:
                            update_archive(x_best, f_best)
                except Exception:
                    pass
                    
        return archive_X
