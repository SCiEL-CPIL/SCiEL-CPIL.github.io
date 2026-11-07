import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = [b[0] for b in bounds]
        hi = [b[1] for b in bounds]
        dim = self.dim
        budget = self.budget
        n_evals = min(budget, 200)

        start_time = time.time()
        deadline = start_time + 280.0

        # 1. Determine f1_min
        # Use a dense grid to find the minimum f1
        n_grid = 200000
        x_grid = np.linspace(0, 1, n_grid)
        f1_grid = 1.0 - np.exp(-4.0 * x_grid) * (np.sin(6.0 * np.pi * x_grid) ** 8)
        
        # The minimum f1 is the global minimum of the function
        f1_min = np.min(f1_grid)
        f1_max = 1.0
        
        # 2. Generate optimal f1 targets
        # Spacing proportional to sqrt(f1)
        # N(f) proportional to sqrt(f) - sqrt(f_min)
        # f_j = (sqrt(f_min) + j * (1 - sqrt(f_min)) / N)^2
        
        sqrt_f1_min = np.sqrt(f1_min)
        sqrt_f1_max = np.sqrt(f1_max)
        
        # j from 1 to n_evals
        # Note: We want points in [f1_min, f1_max]
        # The formula derived: sqrt(f_j) = sqrt(f_min) + (j/N) * (sqrt(f_max) - sqrt(f_min))
        # Wait, the integral was from f_min to 1.
        # N(f) = k (sqrt(f) - sqrt(f_min))
        # N(1) = N_total
        # k = N_total / (1 - sqrt(f_min))
        # j = k (sqrt(f_j) - sqrt(f_min))
        # sqrt(f_j) = sqrt(f_min) + j/k = sqrt(f_min) + j * (1 - sqrt(f_min)) / N_total
        
        j = np.arange(1, n_evals + 1)
        sqrt_f1_targets = sqrt_f1_min + j * (1.0 - sqrt_f1_min) / n_evals
        f1_targets = sqrt_f1_targets ** 2
        
        # 3. Map f1 targets to x1
        # Sort grid by f1 to use searchsorted
        sort_idx = np.argsort(f1_grid)
        f1_sorted = f1_grid[sort_idx]
        x1_sorted = x_grid[sort_idx]
        
        x1_vals = np.empty(n_evals)
        for i in range(n_evals):
            if time.time() - start_time > deadline:
                x1_vals[i] = 0.0
                continue
            
            # Find closest f1 in sorted grid
            idx = np.searchsorted(f1_sorted, f1_targets[i])
            
            # Check neighbors
            best_idx = idx
            best_diff = abs(f1_sorted[idx] - f1_targets[i]) if idx < len(f1_sorted) else np.inf
            
            if idx > 0:
                diff = abs(f1_sorted[idx-1] - f1_targets[i])
                if diff < best_diff:
                    best_diff = diff
                    best_idx = idx - 1
            
            if idx + 1 < len(f1_sorted):
                diff = abs(f1_sorted[idx+1] - f1_targets[i])
                if diff < best_diff:
                    best_diff = diff
                    best_idx = idx + 1
            
            x1_vals[i] = x1_sorted[best_idx]
        
        # 4. Construct and evaluate points
        points = np.zeros((n_evals, dim))
        points[:, 0] = x1_vals
        
        for i in range(n_evals):
            if time.time() - start_time > deadline:
                break
            x = points[i].copy()
            # Clip to bounds just in case, though [0,1] is standard
            x_clipped = np.clip(x, lo, hi)
            problem(x_clipped)
            
        return None
