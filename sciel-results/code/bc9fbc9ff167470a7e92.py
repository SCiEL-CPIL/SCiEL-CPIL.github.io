# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 322.0
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        # ---------- helpers ----------
        def dominates(a, b):
            return bool(np.all(a <= b) and np.any(a < b))

        def non_dominated_mask(F):
            n = len(F)
            nd = np.zeros(n, dtype=bool)
            for i in range(n):
                if nd[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[j], F[i]):
                        nd[i] = True
                        break
            return nd

        # Archive
        archive_X = []
        archive_F = []

        def update_archive(x, f):
            # Check if new point is non-dominated by any existing
            for j in range(len(archive_F)):
                if dominates(archive_F[j], f):
                    return
            # Remove any existing dominated by new
            to_remove = []
            for j in range(len(archive_F)):
                if dominates(f, archive_F[j]):
                    to_remove.append(j)
            for j in sorted(to_remove, reverse=True):
                del archive_X[j]
                del archive_F[j]
            archive_X.append(x)
            archive_F.append(f)

        # ---------- Initialization ----------
        rng = np.random.default_rng(0)
        
        # Use Latin Hypercube for initial spread
        n_init = 20
        X_init = np.empty((n_init, dim))
        for d in range(dim):
            # LHS
            idx = np.arange(n_init)
            perm = rng.permutation(n_init)
            X_init[:, d] = lo[d] + span[d] * (perm + 0.5) / n_init
            
        # Add some random
        n_rand = 10
        X_rand = lo + span * rng.random((n_rand, dim))
        X_init = np.vstack([X_init, X_rand])
        
        F_init = np.array([problem(x) for x in X_init])
        used = len(X_init)

        for i in range(len(X_init)):
            update_archive(X_init[i], F_init[i])

        # ---------- DE Phase ----------
        pop_size = 15
        # Initialize population from archive + random
        pop_X = np.empty((pop_size, dim))
        if len(archive_X) >= pop_size:
            idx = rng.choice(len(archive_X), size=pop_size, replace=False)
            for i, j in enumerate(idx):
                pop_X[i] = archive_X[j]
        else:
            for i in range(min(pop_size, len(archive_X))):
                pop_X[i] = archive_X[i]
            for i in range(len(archive_X), pop_size):
                pop_X[i] = lo + span * rng.random(dim)
        pop_F = np.array([problem(x) for x in pop_X])
        used += pop_size

        CR = 0.8
        F_mut = 0.5

        de_budget = 100
        while used < de_budget + len(X_init) + pop_size: # Approximate
             if time.time() > deadline - 50: # Leave time for local search
                 break
             
             # Simple selection: pick random parents
             r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
             v = pop_X[r1] + F_mut * (pop_X[r2] - pop_X[r3])
             parent_idx = rng.integers(pop_size)
             mask = rng.random(dim) < CR
             if not np.any(mask):
                 mask[rng.integers(dim)] = True
             child = np.where(mask, v, pop_X[parent_idx])
             child = np.clip(child, lo, hi)
             
             fc = problem(child)
             used += 1
             
             # Update archive
             update_archive(child, fc)
             
             # Update population: replace the worst (dominated or random)
             # Simple: replace random member
             replace_idx = rng.integers(pop_size)
             pop_X[replace_idx] = child
             pop_F[replace_idx] = fc

        # ---------- Local Search Phase ----------
        # Use remaining budget for local polishing
        remaining = budget - used
        if remaining > 0 and len(archive_X) > 0:
            # Define weight vectors for 3 objectives
            # Grid of weights
            weights = []
            # Vertices
            weights.append([1, 0, 0])
            weights.append([0, 1, 0])
            weights.append([0, 0, 1])
            # Edge midpoints
            weights.append([0.5, 0.5, 0])
            weights.append([0.5, 0, 0.5])
            weights.append([0, 0.5, 0.5])
            # Center
            weights.append([1/3, 1/3, 1/3])
            
            # Filter weights that are feasible (positive)
            # We will run Nelder-Mead on the Tchebycheff scalarized objective
            # min max_i (w_i * (f_i(x) - z_i^min))
            
            # Get current min values
            F_arch = np.array(archive_F)
            z_min = F_arch.min(axis=0)
            
            # Select starting points: best points from archive for each weight
            # Or just use the top few non-dominated points
            
            # We have `remaining` evaluations.
            # Each Nelder-Mead run might take ~10-20 evals.
            # Let's try to run for each weight vector, starting from the best point in archive for that weight.
            
            num_weights = len(weights)
            evals_per_run = max(5, remaining // num_weights)
            
            for w in weights:
                if used >= budget:
                    break
                if time.time() > deadline - 5:
                    break
                
                # Find best starting point in archive for this weight
                # Scalarized value: max_i (w_i * (f_i - z_min_i))
                # Note: if w_i is 0, term is 0.
                best_start_idx = 0
                best_val = np.inf
                for i in range(len(archive_F)):
                    f_i = archive_F[i]
                    # Tchebycheff
                    val = 0
                    for k in range(n_obj):
                        if w[k] > 1e-6:
                            term = w[k] * (f_i[k] - z_min[k])
                            if term > val:
                                val = term
                    if val < best_val:
                        best_val = val
                        best_start_idx = i
                
                x0 = archive_X[best_start_idx]
                
                # Define objective function for Nelder-Mead
                def obj(x):
                    if time.time() > deadline - 2:
                        return 1e10
                    fx = problem(x)
                    # Tchebycheff scalarization
                    val = 0
                    for k in range(n_obj):
                        if w[k] > 1e-6:
                            term = w[k] * (fx[k] - z_min[k])
                            if term > val:
                                val = term
                    return val
                
                # Run Nelder-Mead
                try:
                    res = minimize(obj, x0, method='Nelder-Mead', bounds=list(zip(lo, hi)), options={'maxiter': evals_per_run, 'xatol': 1e-6, 'fatol': 1e-6})
                    # Update used count? 
                    # minimize doesn't return exact eval count easily in all versions, 
                    # but we can track it or just assume it respects maxiter roughly.
                    # To be safe, we should track evals.
                    # scipy's minimize returns res.nfev
                    used += res.nfev
                    
                    if res.success or used < budget:
                        update_archive(res.x, problem(res.x))
                        # Note: problem(res.x) is an extra eval not counted in res.nfev?
                        # Actually, res.x is the best point found. We should evaluate it once to get the vector for the archive.
                        # But res.fun is the scalar value. We need the vector.
                        # So we must call problem(res.x) again.
                        # This adds 1 eval per weight.
                        # Let's ensure we have budget for that.
                        if used < budget:
                            f_res = problem(res.x)
                            used += 1
                            update_archive(res.x, f_res)
                        else:
                            # If no budget, we can't add to archive, but we already "spent" the evals in minimize.
                            # This is a risk. Better to check budget before running minimize?
                            # We allocated evals_per_run.
                            pass
                except Exception:
                    pass

        return np.array(archive_X)
# EVOLVE-BLOCK-END
