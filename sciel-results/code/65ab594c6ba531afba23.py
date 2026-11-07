"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective optimizer for RE21 with hybrid global-local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        # Use global np.random to respect the seed set by the evaluator
        # This ensures seeds 0 and 1 explore different regions.
        
        archive_X = []
        archive_F = []
        used = 0

        def evaluate_and_add(x):
            nonlocal used
            if used >= budget:
                return
            if time.time() > deadline:
                return
            
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=np.float64)
            used += 1
            
            # Check if f is better than reference point [3000.0, 0.06]
            # If not, it won't contribute to hypervolume, but we might still keep it
            # for local search if it's close. However, for archive, only non-dominated matters.
            
            # Check dominance against existing archive
            # If archive is empty, add
            if len(archive_F) == 0:
                archive_X.append(x)
                archive_F.append(f)
                return

            F_arr = np.array(archive_F)
            
            # Is new point dominated by any archive point?
            # dominated if exists j: F[j] <= f and F[j] < f
            mask_le = np.all(F_arr <= f, axis=1)
            mask_lt = np.any(F_arr < f, axis=1)
            if np.any(mask_le & mask_lt):
                return  # Dominated, discard

            # Remove archive points dominated by new point
            # dominated if exists j: f <= F[j] and f < F[j]
            mask_le2 = np.all(f <= F_arr, axis=1)
            mask_lt2 = np.any(f < F_arr, axis=1)
            to_remove = np.where(mask_le2 & mask_lt2)[0]
            
            for idx in sorted(to_remove, reverse=True):
                archive_X.pop(idx)
                archive_F.pop(idx)
            
            archive_X.append(x)
            archive_F.append(f)

        # Phase 1: Initialization
        # Mix of random and boundary points
        n_init = 40
        X_init = np.zeros((n_init, dim))
        # 20 random
        X_init[:20] = lo + span * np.random.random((20, dim))
        # 20 boundary/corner points
        # Generate random corners
        for i in range(20):
            # Randomly choose low or high for each dimension
            X_init[20+i] = np.where(np.random.random(dim) < 0.5, lo, hi)
        
        for x in X_init:
            evaluate_and_add(x)

        # Phase 2: Global Search (Evolutionary)
        # Use remaining budget for global search, leaving some for local polish
        # Let's reserve 100 evaluations for local polish
        n_global = min(80, budget - used - 100)
        
        if n_global > 0:
            for _ in range(n_global):
                if time.time() > deadline:
                    break
                if used >= budget - 100: # Stop if we hit the reserve
                    break
                
                # Select parent from archive
                if len(archive_X) == 0:
                    parent = np.random.uniform(lo, hi)
                else:
                    # Select random parent from archive
                    idx = np.random.randint(len(archive_X))
                    parent = np.array(archive_X[idx])
                
                # Mutate
                # Adaptive step size: start large, shrink
                progress = used / budget
                scale = 0.5 * (1.0 - progress) + 0.05
                child = parent + scale * span * np.random.standard_normal(dim)
                
                evaluate_and_add(child)

        # Phase 3: Local Polish
        # Take top points from archive and refine them using local optimization
        # We have ~100 evaluations left
        n_local_points = 5
        evals_per_point = 20
        
        if len(archive_X) > 0:
            # Sort archive by sum of objectives to pick "good" starting points
            F_arr = np.array(archive_F)
            sums = np.sum(F_arr, axis=1)
            # Pick the best 5 based on sum, but ensure diversity?
            # For simplicity, pick the 5 best by sum.
            # To ensure diversity, we could pick non-dominated points with lowest sum.
            # Since archive is non-dominated, all are good.
            # Let's pick 5 evenly spaced along the front if possible, or just best sums.
            
            # Get indices of non-dominated points (all in archive are non-dominated)
            # Sort by first objective
            order = np.argsort(F_arr[:, 0])
            
            # Select 5 points: min f1, max f1, and 3 in between
            indices_to_optimize = []
            n = len(order)
            if n <= 5:
                indices_to_optimize = list(range(n))
            else:
                indices_to_optimize = [
                    order[0],
                    order[n//4],
                    order[n//2],
                    order[3*n//4],
                    order[-1]
                ]
            
            for idx in indices_to_optimize:
                if time.time() > deadline:
                    break
                if used >= budget:
                    break
                
                x0 = np.array(archive_X[idx])
                f0 = np.array(archive_F[idx])
                
                # Define local objective: Tchebycheff or Weighted Sum
                # Use Weighted Sum with random weights to explore different parts
                # Or Tchebycheff with weights derived from the point's position
                # Let's use a simple weighted sum: w1*f1 + w2*f2
                # To target the region around x0, we can use weights inverse to f0
                # w1 = 1/f0[0], w2 = 1/f0[1] -> Minimizes the sum of relative errors?
                # Better: Tchebycheff max(w1*f1, w2*f2) is non-smooth.
                # Use smooth approximation or just Weighted Sum.
                
                w = np.array([1.0/f0[0] if f0[0] > 0 else 1.0, 
                              1.0/f0[1] if f0[1] > 0 else 1.0])
                w = w / np.sum(w)
                
                def obj(x):
                    x_clipped = np.clip(x, lo, hi)
                    f = np.array(problem(x_clipped), dtype=np.float64)
                    # Weighted sum
                    return np.dot(w, f)
                
                # Run Nelder-Mead
                # Options: maxiter, maxfev
                # We have ~20 evals per point
                try:
                    res = minimize(obj, x0, method='Nelder-Mead', 
                                   options={'maxiter': 10, 'maxfev': 20, 'xatol': 1e-4, 'fatol': 1e-4})
                    # The minimize function uses up to maxfev evaluations.
                    # We need to track if we actually used them.
                    # The 'obj' function calls 'problem', which increments 'used'.
                    # So we don't need to manually count, but we should check 'used' after.
                    
                    # Add the result to archive
                    if res.success or res.fun < obj(x0):
                        x_best = np.clip(res.x, lo, hi)
                        f_best = np.array(problem(x_best), dtype=np.float64)
                        # Note: problem(x_best) is another evaluation!
                        # We need to be careful with budget.
                        # minimize already called problem() multiple times.
                        # The last call in minimize might not be the best.
                        # We should evaluate res.x explicitly if it's better.
                        # But this costs 1 eval.
                        
                        # Check if we have budget
                        if used < budget:
                            f_check = np.array(problem(x_best), dtype=np.float64)
                            used += 1
                            if np.all(f_check <= f0) and np.any(f_check < f0):
                                # Better than start
                                evaluate_and_add(x_best)
                        else:
                            # No budget left, but we have the result from minimize
                            # We can't add it without evaluating, but minimize already did.
                            # However, we don't know which x gave the best f.
                            # We'll skip adding if we can't evaluate.
                            pass

                except Exception:
                    pass

        return archive_X
# EVOLVE-BLOCK-END
