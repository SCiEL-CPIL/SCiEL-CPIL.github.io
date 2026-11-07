import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        # Strategy:
        # 1. Sample the front with a coarse grid (64 points)
        # 2. Use remaining budget for local refinement on non-dominated points
        
        n_init = 64
        X_init = np.zeros((n_init, dim))
        
        # Grid: 4x4x4 = 64 points in first 3 variables
        g1 = np.linspace(0, 1, 4)
        g2 = np.linspace(0, 1, 4)
        g3 = np.linspace(0, 1, 4)
        
        count = 0
        for a in g1:
            for b in g2:
                for c in g3:
                    if count >= n_init:
                        break
                    x = np.zeros(dim)
                    x[0] = lo[0] + a * (hi[0] - lo[0])
                    x[1] = lo[1] + b * (hi[1] - lo[1])
                    x[2] = lo[2] + c * (hi[2] - lo[2])
                    # Set remaining variables to mid-range
                    x[3:] = (lo[3:] + hi[3:]) / 2.0
                    X_init[count] = x
                    count += 1
        
        F_init = np.array([problem(x) for x in X_init])
        used = n_init
        
        # Find non-dominated points
        def is_non_dominated(F, idx):
            for j in range(len(F)):
                if j == idx:
                    continue
                if np.all(F[j] <= F[idx]) and np.any(F[j] < F[idx]):
                    return False
            return True
        
        nondom_indices = [i for i in range(len(F_init)) if is_non_dominated(F_init, i)]
        
        # Select up to 16 non-dominated points for refinement
        if len(nondom_indices) > 16:
            sorted_idx = np.argsort(F_init[nondom_indices, 0])
            step = max(1, len(sorted_idx) // 16)
            selected = sorted_idx[::step][:16]
            search_indices = [nondom_indices[i] for i in selected]
        else:
            search_indices = nondom_indices
        
        remaining_budget = budget - used
        if remaining_budget <= 0:
            return X_init
        
        n_search = len(search_indices)
        if n_search == 0:
            return X_init
            
        evals_per_search = max(10, remaining_budget // n_search)
        
        from scipy.optimize import minimize
        
        for idx in search_indices:
            if time.time() > deadline:
                break
            if used >= budget:
                break
                
            x0 = X_init[idx].copy()
            
            def obj_func(x):
                nonlocal used
                if used >= budget:
                    return np.sum(F_init[idx])
                if time.time() > deadline:
                    return np.sum(F_init[idx])
                
                x_clipped = np.clip(x, lo, hi)
                f = problem(x_clipped)
                used += 1
                return np.sum(f)
                
            try:
                res = minimize(obj_func, x0, method='Nelder-Mead', options={'maxiter': evals_per_search, 'xatol': 1e-6, 'fatol': 1e-6})
                X_init[idx] = res.x
                if used < budget and time.time() < deadline:
                    F_init[idx] = problem(res.x)
                    used += 1
            except:
                pass
        
        return X_init
