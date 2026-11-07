import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0
        
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

        # 1. Initialization with Latin Hypercube Sampling
        n_init = 50
        # LHS for better spread
        X_init = np.zeros((n_init, dim))
        for d in range(dim):
            # Divide [0,1] into n_init intervals, pick one point in each
            intervals = np.random.random(n_init)
            # Shift each to its interval
            X_init[:, d] = (np.arange(n_init) + intervals) / n_init
        X_init = lo + span * X_init
        
        for x in X_init:
            f = evaluate(x)
            if f is not None:
                update_archive(x, f)
        
        # 2. Differential Evolution Phase
        # We maintain a population of size N_pop
        # N_pop should be at least 5 * dim for DE
        N_pop = 20
        
        # Initialize population from archive + random
        # If archive is small, pad with random
        pop_X = []
        pop_F = []
        
        # Add archive points
        for x, f in zip(archive_X, archive_F):
            if len(pop_X) < N_pop:
                pop_X.append(x)
                pop_F.append(f)
        
        # Pad with random points
        while len(pop_X) < N_pop:
            x = np.random.uniform(lo, hi)
            f = evaluate(x)
            if f is None:
                break
            pop_X.append(x)
            pop_F.append(f)
            
        # DE Loop
        # Budget remaining for DE: budget - n_init - local_polish_budget
        local_polish_budget = 40
        de_budget = budget - n_init - local_polish_budget
        
        # DE Parameters
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate
        
        while evals_used < budget - local_polish_budget and time.time() < deadline:
            # Generate N_pop trial vectors
            trial_X = []
            trial_F = []
            
            for i in range(N_pop):
                if evals_used >= budget - local_polish_budget or time.time() > deadline:
                    break
                    
                # Select 3 distinct individuals
                j, k, l = np.random.choice(len(pop_X), 3, replace=False)
                
                # Differential mutation
                v = pop_X[j] + F * (pop_X[k] - pop_X[l])
                
                # Binomial crossover
                r = np.random.randint(dim)
                for d in range(dim):
                    if np.random.random() < CR or d == r:
                        v[d] = v[d]
                    else:
                        v[d] = pop_X[i][d]
                        
                v = np.clip(v, lo, hi)
                f = evaluate(v)
                if f is None:
                    break
                trial_X.append(v)
                trial_F.append(f)
                
                # Update archive immediately? No, let's do selection first.
                # But we can update archive to keep it fresh.
                # Actually, updating archive inside the loop might be expensive.
                # Let's just collect them and do selection.
                
            if not trial_X:
                break
                
            # Combine population and trial vectors
            combined_X = pop_X + trial_X
            combined_F = pop_F + trial_F
            
            # Non-dominated sorting + Hypervolume contribution selection
            # Simplified: Select N_pop points that maximize hypervolume contribution
            # Or just use non-dominated sorting and crowding distance.
            # For small N_pop, we can do exact HV selection.
            
            # Let's use a simple greedy HV maximization on the combined set
            # to select the next population.
            # This is O(N^2 * d) which is fine for N=40.
            
            # First, filter out dominated points from the combined set
            # We want the next N_pop to be non-dominated and spread out.
            
            # 1. Find non-dominated points in combined set
            non_dom_idx = []
            for i in range(len(combined_X)):
                is_dominated = False
                for j in range(len(combined_X)):
                    if i == j:
                        continue
                    if dominates(combined_F[j], combined_F[i]):
                        is_dominated = True
                        break
                if not is_dominated:
                    non_dom_idx.append(i)
                    
            if len(non_dom_idx) == 0:
                # Fallback: keep best N_pop by first objective
                non_dom_idx = np.argsort(combined_F, axis=None)[:N_pop].tolist()
                
            # If we have more non-dominated points than N_pop, select N_pop
            if len(non_dom_idx) > N_pop:
                # Greedy HV selection
                selected = []
                remaining = non_dom_idx.copy()
                
                # Reference point for HV (local, just for selection)
                # Use current archive nadir + margin
                if archive_F:
                    z_max = np.max(archive_F, axis=0) * 1.1 + 0.1
                else:
                    z_max = np.array([100, 1.0])
                    
                while len(selected) < N_pop and remaining:
                    best_idx = None
                    best_hv = -1
                    for idx in remaining:
                        # Calculate HV contribution of this point given 'selected'
                        # This is expensive. Approximation: pick point with best "angle" or spread.
                        # Let's use a simpler heuristic: pick point that is farthest from existing selected points in objective space.
                        if selected:
                            F_sel = np.array([combined_F[s] for s in selected])
                            dists = np.linalg.norm(F_sel - combined_F[idx], axis=1)
                            score = np.min(dists)
                        else:
                            score = 0
                            
                        if score > best_hv:
                            best_hv = score
                            best_idx = idx
                            
                    selected.append(best_idx)
                    remaining.remove(best_idx)
                    
                final_indices = selected
            else:
                final_indices = non_dom_idx
                
            # Update population
            pop_X = [combined_X[i] for i in final_indices]
            pop_F = [combined_F[i] for i in final_indices]
            
            # Update global archive with new population points
            for x, f in zip(pop_X, pop_F):
                update_archive(x, f)

        # 3. Local Polish
        # Select diverse points from archive and use diverse weight vectors
        if len(archive_X) > 0:
            F_arr = np.array(archive_F)
            # Sort by first objective to ensure we pick across the front
            order = np.argsort(F_arr[:, 0])
            
            # Select up to 10 points, evenly spaced in index
            n_points_to_refine = min(10, len(archive_X))
            # Pick indices
            indices = np.linspace(0, len(order)-1, n_points_to_refine).astype(int)
            
            # Create diverse weight vectors for Tchebycheff decomposition
            weights_list = []
            for i in range(n_points_to_refine):
                # Map index to weight: early indices -> high weight on f1, late -> high weight on f2
                w1 = 1.0 - (i / max(1, n_points_to_refine - 1)) * 0.9  # w1 from 1.0 to 0.1
                w2 = 1.0 - w1
                weights_list.append(np.array([w1, w2]))
            
            # Nadir point estimate: use max of current archive with some margin
            z_max = np.max(F_arr, axis=0) if len(F_arr) > 0 else np.ones(n_obj)
            # Add small margin to avoid issues
            z_max = z_max * 1.01 + 1e-6
            
            for idx in indices:
                if evals_used >= budget or time.time() > deadline:
                    break
                
                x0 = np.array(archive_X[order[idx]])
                f0 = np.array(archive_F[order[idx]])
                
                # Use the corresponding weight vector
                w = weights_list[idx]
                
                # Use soft-max (log-sum-exp) for smoother objective
                beta = 50.0  # Temperature parameter for soft-max
                
                def obj(x):
                    x_clipped = np.clip(x, lo, hi)
                    f = np.array(problem(x_clipped), dtype=np.float64)
                    # Tchebycheff with soft-max: -log(sum(exp(-beta * w * (f - z_max))))
                    # This is a smooth approximation of max(w * (f - z_max))
                    vals = w * (f - z_max)
                    # Use log-sum-exp for numerical stability
                    max_val = np.max(vals)
                    if max_val > 1e10:  # Numerical safety
                        return max_val
                    log_sum = max_val + np.log(np.sum(np.exp(-beta * (vals - max_val))))
                    return log_sum / beta  # Scale back to original scale
                    
                # Note: This objective function calls problem, which increments evals_used.
                # We need to ensure we don't exceed budget.
                # minimize will call obj multiple times.
                
                # We have ~40 evals left total. 10 points * ~4 evals each = 40.
                # Set maxfev=4 to be safe.
                
                try:
                    res = minimize(obj, x0, method='Nelder-Mead', 
                                   options={'maxiter': 10, 'maxfev': 4, 'xatol': 1e-5, 'fatol': 1e-5})
                    # Check if res.x is better than x0 in some sense
                    # We need to evaluate res.x explicitly if we want to add it to archive?
                    # minimize already evaluated it? Not necessarily the best one in the final state.
                    # We must evaluate res.x to get its true F.
                    if evals_used < budget:
                        x_best = np.clip(res.x, lo, hi)
                        f_best = evaluate(x_best)
                        if f_best is not None:
                            update_archive(x_best, f_best)
                except Exception:
                    pass
                    
        return archive_X

# Import minimize at the top level to avoid local scope issues or delays
from scipy.optimize import minimize
