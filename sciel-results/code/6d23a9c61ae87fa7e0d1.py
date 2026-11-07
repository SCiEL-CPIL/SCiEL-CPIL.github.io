import time
import numpy as np
import random

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # We will use a subset of the budget for initialization and local search
        # ZDT4 is deceptive, so we initialize x[1:] close to 0
        # x[0] is the shape variable, so we spread it in [0, 1] (or bounds)
        
        # Note: bounds[0] might be [-5, 5] or [0, 1] depending on the specific ZDT4 variant.
        # Standard ZDT4: x1 in [0, 1], x2..xn in [-5, 5].
        # But we must respect `bounds` provided.
        
        # Let's assume the "good" region for x1 is the full range, but for x2..xn it is around 0.
        
        pop_size = 20
        # Adjust pop_size if budget is small
        if self.budget < 50:
             pop_size = max(5, self.budget // 5)
        
        archive = []
        used_evals = 0
        
        # 1. Initialization
        # x1: uniform in bounds
        # x2..xn: uniform in a small range around 0, clipped to bounds
        init_x1 = np.random.uniform(lo[0], hi[0], pop_size)
        
        # For x2..xn, we want them near 0.
        # Let's take a small fraction of the range around 0.
        # If bounds are symmetric, e.g. [-5, 5], use [-0.5, 0.5].
        # If bounds are [0, 1], use [0, 0.1] or [0.4, 0.6]?
        # ZDT4 usually has global min at 0. If bounds are [0, 1], 0 is on the edge.
        # Let's just use a small range around 0, clipped.
        
        # To be safe and general:
        # If 0 is in bounds, use a small interval around 0.
        # If 0 is not in bounds, use the center of the bounds?
        # For ZDT4, the "trap" is that local minima exist away from 0.
        # If bounds are [0, 1], the global min is at 0 (boundary).
        # If bounds are [-5, 5], global min is at 0.
        
        # Strategy: Initialize x[1:] with values drawn from a distribution centered at 0, 
        # with a scale that is a small fraction of the range.
        
        centers = np.zeros(self.dim)
        scales = (hi - lo) * 0.05 # 5% of range
        
        X = []
        F = []
        
        for i in range(pop_size):
            x = np.zeros(self.dim)
            x[0] = init_x1[i]
            # For other dims, sample from normal(0, scale) clipped to bounds
            # Or just uniform in [center - scale, center + scale]
            for j in range(1, self.dim):
                # If 0 is within bounds, center at 0.
                # If not, center at the bound closest to 0?
                # Standard ZDT4 x2..xn are [-5, 5].
                c = 0.0 if lo[j] <= 0.0 <= hi[j] else (lo[j] + hi[j]) / 2.0
                s = max(0.1, (hi[j] - lo[j]) * 0.05)
                x[j] = np.random.uniform(c - s, c + s)
                x[j] = np.clip(x[j], lo[j], hi[j])
            
            fx = problem(x)
            used_evals += 1
            X.append(x)
            F.append(fx)
            archive.append(fx)
            
            if time.time() > deadline:
                return archive
            
            if used_evals >= self.budget:
                break

        # 2. Local Search / Polishing
        # We have remaining_budget = self.budget - used_evals
        # We want to polish the points that look good (low f1) or all of them.
        # Let's sort by f1 and polish the best ones.
        
        # Create indices sorted by f1
        indices = np.argsort([f[0] for f in F])
        
        # Distribute remaining budget among the top K points
        # Let's try to give each of the top 10 points some steps.
        # Or just iterate through the population and do 1 step each until budget runs out.
        
        # A more effective approach for ZDT4:
        # The "trap" is that f1 has local minima.
        # Moving towards 0 reduces f1.
        # Let's do a coordinate descent towards 0 for the most promising points.
        
        # We'll pick the top 10 points by f1.
        top_k = min(10, len(X))
        top_indices = indices[:top_k]
        
        steps_per_point = max(1, (self.budget - used_evals) // top_k)
        
        for idx in top_indices:
            if used_evals >= self.budget:
                break
            if time.time() > deadline:
                break
                
            x = X[idx].copy()
            f = F[idx].copy()
            
            # Perform steps
            for _ in range(steps_per_point):
                if used_evals >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Pick a random dimension to improve (1 to dim-1)
                # We focus on dimensions that are not 0 (the shape variable)
                dim_idx = np.random.randint(1, self.dim)
                
                # Propose moving halfway towards 0
                # If x[dim_idx] is 0, move slightly away? No, 0 is the target.
                # If x[dim_idx] is not 0, move towards 0.
                target = 0.0
                # Clip target to bounds
                if target < lo[dim_idx]: target = lo[dim_idx]
                if target > hi[dim_idx]: target = hi[dim_idx]
                
                # Step size: move 50% of the distance
                new_val = x[dim_idx] + 0.5 * (target - x[dim_idx])
                
                x_new = x.copy()
                x_new[dim_idx] = new_val
                x_new = np.clip(x_new, lo, hi)
                
                f_new = problem(x_new)
                used_evals += 1
                archive.append(f_new)
                
                # Compare f1 (first objective)
                if f_new[0] < f[0]:
                    x = x_new
                    f = f_new
                    X[idx] = x
                    F[idx] = f
                # Else, keep old x, f
                
        return archive
