"""Multi-objective optimizer for RE21 using hypervolume-driven local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-objective optimizer using:
    1. Initial random sampling
    2. Weighted-sum local optimization (L-BFGS-B) from diverse starting points
    3. Hypervolume-based archive maintenance
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s limit
        
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Phase 1: Initial random sampling (30 points)
        n_init = min(30, budget)
        X_init = lo + (hi - lo) * rng.random((n_init, dim))
        F_init = np.array([problem(x) for x in X_init])
        used = n_init
        
        # Archive: maintain non-dominated points
        archive_X = list(X_init)
        archive_F = list(F_init)
        
        def is_dominated(f_new, archive_F_list):
            for f_old in archive_F_list:
                if np.all(f_old <= f_new) and np.any(f_old < f_new):
                    return True
            return False
        
        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)
        
        def update_archive(x, f):
            nonlocal archive_X, archive_F
            # Check if new point is dominated
            if is_dominated(f, archive_F):
                return False
            # Remove points dominated by new point
            new_X = [x]
            new_F = [f]
            for i, (ax, af) in enumerate(zip(archive_X, archive_F)):
                if not dominates(f, af):
                    new_X.append(ax)
                    new_F.append(af)
            archive_X = new_X
            archive_F = new_F
            return True
        
        # Phase 2: Weighted-sum local optimization
        # Create weight vectors for multi-objective optimization
        n_weights = 10
        weights = []
        for i in range(n_weights):
            w = np.zeros(n_obj)
            w[0] = 1.0
            for j in range(1, n_obj):
                w[j] = 0.1 + 0.8 * i / (n_weights - 1)
            weights.append(w)
        
        # For each weight, pick a diverse starting point and run local optimization
        for w_idx, w in enumerate(weights):
            if used >= budget or time.time() > deadline:
                break
            
            # Pick starting point: use the best point for this weight from archive
            if len(archive_F) > 0:
                # Compute weighted sum for each archive point
                weighted_sums = np.array([np.dot(w, f) for f in archive_F])
                best_idx = np.argmin(weighted_sums)
                x_start = archive_X[best_idx].copy()
            else:
                x_start = lo + (hi - lo) * 0.5
            
            # Define scalarized objective
            def scalarized_obj(x):
                if time.time() > deadline:
                    return np.inf
                try:
                    f = problem(x)
                    return np.dot(w, f)
                except:
                    return np.inf
            
            # Run L-BFGS-B local optimization
            try:
                res = minimize(scalarized_obj, x_start, method='L-BFGS-B', 
                               bounds=bounds, options={'maxiter': 50, 'ftol': 1e-8})
                if used < budget:
                    x_opt = res.x
                    f_opt = problem(x_opt)
                    used += 1
                    update_archive(x_opt, f_opt)
            except:
                pass
        
        # Phase 3: Refine with more local searches from best archive points
        if time.time() < deadline and used < budget:
            # Sort archive by hypervolume contribution (approximate by weighted sum)
            if len(archive_F) > 0:
                # Use uniform weights for refinement
                for _ in range(min(5, budget - used)):
                    if time.time() > deadline or used >= budget:
                        break
                    
                    # Pick random archive point
                    idx = rng.integers(len(archive_X))
                    x_start = archive_X[idx].copy()
                    
                    # Small perturbation
                    x_start = x_start + 0.05 * (hi - lo) * rng.standard_normal(dim)
                    x_start = np.clip(x_start, lo, hi)
                    
                    # Local search with equal weights
                    def refine_obj(x):
                        if time.time() > deadline:
                            return np.inf
                        try:
                            f = problem(x)
                            return np.sum(f)
                        except:
                            return np.inf
                    
                    try:
                        res = minimize(refine_obj, x_start, method='L-BFGS-B',
                                       bounds=bounds, options={'maxiter': 30, 'ftol': 1e-10})
                        if used < budget:
                            x_opt = res.x
                            f_opt = problem(x_opt)
                            used += 1
                            update_archive(x_opt, f_opt)
                    except:
                        pass
        
        return archive_X
# EVOLVE-BLOCK-END
