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
        span = hi - lo
        mid = (lo + hi) / 2.0
        
        # WFG7 specific strategy:
        # 1. The Pareto front is a portion of a unit sphere in the positive orthant of objective space.
        # 2. The first 3 variables (typically) control the position on the front (angles).
        # 3. The remaining variables control the distance from the front (deviation).
        # 4. We want to set distance variables to minimize deviation (often 0 or 0.5 depending on bounds)
        #    and vary the position variables to cover the front.
        
        # Assume first 3 variables are position, rest are distance.
        n_pos = 3
        n_dist = dim - n_pos
        
        # Identify "good" values for distance variables.
        # In many WFG7 implementations, distance variables are in [0, 1] and 0 or 0.5 minimizes deviation.
        # We will try 0, 0.5, and 1.0 for distance variables, but focus on 0 and 0.5.
        
        # Phase 1: Sample the front with a grid in the position space.
        # We have 200 evaluations.
        # Let's use 100 evaluations for a coarse grid of the front (position vars varied, dist vars fixed).
        # Then 100 evaluations for local refinement.
        
        # Grid size for 3D position space: 10x10x10 = 1000 is too many.
        # 4x4x4 = 64 points.
        # Let's use a spherical coordinate parameterization for the first 3 variables.
        # Or just a simple grid in the first 3 variables.
        
        # Let's assume the first 3 variables map to angles.
        # We will sample a set of points on the front.
        
        # Strategy:
        # 1. Generate a set of candidate points on the front by varying the first 3 variables.
        # 2. Fix the remaining variables to 0 (or mid-range).
        # 3. Evaluate these points.
        # 4. Use the remaining budget to perform local search (Nelder-Mead or similar) around the best points to push them closer to the ideal front.
        
        # Let's define a function to generate points on the front.
        # For WFG7, the position variables are often [0, 1].
        
        # Let's use a set of points that cover the sphere.
        # We can use Fibonacci sphere or just a grid.
        
        # Since we don't know the exact mapping, we will sample the first 3 variables uniformly.
        
        n_init = 60
        X_init = np.zeros((n_init, dim))
        
        # Vary first 3 variables in a grid
        # 4x4x4 = 64 points, we use 60 to leave room for randoms if needed, but let's do 4x4x4
        # Actually, let's use a mix of grid and random for better coverage.
        # 30 grid points, 30 random points.
        
        n_grid = 30
        n_rand = n_init - n_grid
        
        # Grid points
        # 3x3x3 = 27 points
        g1 = np.linspace(0, 1, 3)
        g2 = np.linspace(0, 1, 3)
        g3 = np.linspace(0, 1, 3)
        
        count = 0
        for a in g1:
            for b in g2:
                for c in g3:
                    if count >= n_grid:
                        break
                    x = np.zeros(dim)
                    x[0] = lo[0] + a * (hi[0] - lo[0])
                    x[1] = lo[1] + b * (hi[1] - lo[1])
                    x[2] = lo[2] + c * (hi[2] - lo[2])
                    # Set distance variables to 0.5 (mid-range) as a heuristic for WFG7
                    x[n_pos:] = mid[n_pos:]
                    X_init[count] = x
                    count += 1
        
        # Random points
        # Use np.random for reproducibility with seed
        for i in range(n_grid, n_init):
            x = np.zeros(dim)
            # Random position variables
            x[:n_pos] = lo[:n_pos] + np.random.rand(n_pos) * span[:n_pos]
            # Random distance variables, but bias towards mid?
            # Let's just use uniform
            x[n_pos:] = lo[n_pos:] + np.random.rand(n_dist) * span[n_pos:]
            X_init[i] = x
        
        F_init = np.array([problem(x) for x in X_init])
        used = n_init
        
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
        
        # If we have fewer than 10 non-dominated points, we might need to add more random points.
        # But let's assume the grid covers it reasonably.
        
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
