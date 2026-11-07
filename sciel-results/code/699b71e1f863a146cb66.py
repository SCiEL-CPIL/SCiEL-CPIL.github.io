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
        
        # WFG7 specific strategy:
        # The Pareto front is a portion of a unit sphere in the positive orthant of objective space.
        # Variables 0, 1, 2 typically control position on the front.
        # Variables 3-11 control deviation from the front.
        
        n_pos = 3
        n_dist = dim - n_pos
        
        # Heuristic: For WFG7, setting deviation variables to 0.5 often minimizes deviation.
        # We will fix deviation vars to 0.5 and vary position vars.
        
        # Phase 1: Sample the front with a grid in the position space.
        # We have 200 evaluations.
        # Let's use 64 evaluations for a 4x4x4 grid of the front.
        # Then 136 evaluations for local refinement (Nelder-Mead).
        
        n_grid = 64
        X_init = np.zeros((n_grid, dim))
        
        # Grid points for position variables (0, 1, 2)
        g = np.linspace(0, 1, 4)
        
        count = 0
        for a in g:
            for b in g:
                for c in g:
                    x = np.zeros(dim)
                    x[0] = lo[0] + a * (hi[0] - lo[0])
                    x[1] = lo[1] + b * (hi[1] - lo[1])
                    x[2] = lo[2] + c * (hi[2] - lo[2])
                    # Set distance variables to 0.5 (mid-range) as a heuristic for WFG7
                    # Assuming bounds are [0, 1] for simplicity, but using mid for generality
                    x[n_pos:] = (lo[n_pos:] + hi[n_pos:]) / 2.0
                    X_init[count] = x
                    count += 1
        
        F_init = np.array([problem(x) for x in X_init])
        used = n_grid
        
        # Now we have a set of points. Some might be dominated.
        # We will perform local search on the non-dominated points.
        
        # Find non-dominated points
        def is_non_dominated(F, idx):
            for j in range(len(F)):
                if j == idx:
                    continue
                if np.all(F[j] <= F[idx]) and np.any(F[j] < F[idx]):
                    return False
            return True
        
        nondom_indices = [i for i in range(len(F_init)) if is_non_dominated(F_init, i)]
        
        # Select top K non-dominated points for local search
        # We want to maximize hypervolume, so we want a spread of points.
        # Let's pick points that are spread out in objective space.
        
        # For simplicity, let's just pick all non-dominated points if count <= 20, otherwise sample 20.
        if len(nondom_indices) > 20:
            # Simple spread: sort by first objective, pick every k-th
            sorted_idx = np.argsort(F_init[nondom_indices, 0])
            step = max(1, len(sorted_idx) // 20)
            selected = sorted_idx[::step][:20]
            search_indices = [nondom_indices[i] for i in selected]
        else:
            search_indices = nondom_indices
        
        # Local Search: Nelder-Mead on each point to minimize a scalarized objective?
        # Or just minimize the sum of objectives?
        # Since we want to stay on the front, minimizing the sum might work if the front is convex.
        # WFG7 front is spherical (convex).
        
        # We have budget - used evaluations left.
        # Let's allocate remaining budget to local search.
        
        remaining_budget = budget - used
        if remaining_budget <= 0:
            return X_init
        
        # Number of local searches
        n_search = len(search_indices)
        if n_search == 0:
            return X_init
            
        evals_per_search = max(10, remaining_budget // n_search)
        
        # Perform Nelder-Mead on each selected point
        # Minimize sum of objectives
        
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
                    return np.sum(F_init[idx]) # Return current value if out of budget
                if time.time() > deadline:
                    return np.sum(F_init[idx])
                
                # Clip to bounds
                x_clipped = np.clip(x, lo, hi)
                f = problem(x_clipped)
                used += 1
                return np.sum(f)
                
            # Use Nelder-Mead
            try:
                res = minimize(obj_func, x0, method='Nelder-Mead', options={'maxiter': evals_per_search, 'xatol': 1e-6, 'fatol': 1e-6})
                # Update the point
                X_init[idx] = res.x
                # Re-evaluate to get accurate F
                if used < budget and time.time() < deadline:
                    F_init[idx] = problem(res.x)
                    used += 1
            except:
                pass
        
        return X_init
