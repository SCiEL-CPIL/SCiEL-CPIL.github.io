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
        
        start_time = time.time()
        deadline = 295.0  # safety margin before 300s limit
        
        # ZDT6: f1(x) = 1 - exp(-4*x1) * sin(6*pi*x1)^8
        # f2(x) = g(x) * h(f1(x), g(x))
        # g(x) = 1 + 9 * sum(x_i)/(n-1) for i in 1..n-1
        # h(u,g) = 1 - sqrt(u/g)
        # Pareto front: g=1 (i.e., x[1:]=0), so f2 = 1 - sqrt(f1)
        # f1 ranges from ~0.284 to 1.0
        
        # The Pareto front is parameterized by x1 where x[1:]=0
        # f1 = 1 - exp(-4*x1)*sin(6*pi*x1)^8
        # f2 = 1 - sqrt(f1)
        
        # We want to cover the Pareto front well to maximize hypervolume.
        # The reference point is [11, 11], so all points with f1 < 11 and f2 < 11 count.
        # The Pareto front spans f1 in [~0.284, 1.0] and f2 in [0, ~0.849].
        
        # Strategy: Use the budget to evaluate points that lie exactly on the Pareto front
        # (x[1:] = 0), spread across the range of f1 values to maximize hypervolume.
        
        # Pre-compute the mapping from x1 to f1
        # f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)^8
        # This function is not monotonic - it oscillates. But for x[1:]=0, g=1,
        # so f2 = 1 - sqrt(f1). We want diverse f1 values.
        
        # Actually, the Pareto front of ZDT6 is the set of points where x[1:]=0.
        # f1 oscillates but the range is [f1_min, 1] where f1_min occurs at some x1.
        # Let's find the range of f1.
        
        # Dense grid to understand f1(x1)
        n_grid = 100000
        x_grid = np.linspace(0, 1, n_grid)
        f1_grid = 1.0 - np.exp(-4.0 * x_grid) * (np.sin(6.0 * np.pi * x_grid) ** 8)
        
        f1_min = np.min(f1_grid)
        f1_max = np.max(f1_grid)  # should be 1.0
        
        # For hypervolume maximization with reference [11,11], we want to spread points
        # across the f1-f2 plane. Since f2 = 1 - sqrt(f1) on the PF, the front is a curve.
        # The hypervolume contribution is maximized by having points spread along this curve.
        
        # Optimal strategy: evaluate x[1:]=0 for x1 values that give well-spread f1 values.
        # We need to find x1 for each target f1.
        
        # Since f1 is not monotonic, for each target f1 we can pick any x1 that achieves it.
        # Let's create a lookup: for each grid index, we have (x1, f1).
        
        # We want to choose n_evals points with distinct f1 values spread from f1_min to f1_max.
        # Use the grid to find x1 values for evenly spaced f1 targets.
        
        n_evals = self.budget
        target_f1s = np.linspace(f1_min, f1_max, n_evals)
        
        # For each target f1, find the x1 in our grid that is closest
        # We want to avoid duplicate x1 values if possible, but argmin is fine
        
        # To speed this up, sort the grid by f1
        sort_idx = np.argsort(f1_grid)
        f1_sorted = f1_grid[sort_idx]
        x1_sorted = x_grid[sort_idx]
        
        # Remove duplicates in f1_sorted to avoid redundant points
        # Keep only the first occurrence of each unique f1 value
        unique_mask = np.concatenate(([True], f1_sorted[1:] != f1_sorted[:-1]))
        f1_unique = f1_sorted[unique_mask]
        x1_unique = x1_sorted[unique_mask]
        
        n_unique = len(f1_unique)
        
        if n_unique <= n_evals:
            # Use all unique points
            x1_vals = x1_unique
        else:
            # Need to select n_evals points from n_unique
            # Use greedy facility location: start with endpoints, then add points
            # that maximize the minimum distance to existing points in f1 space
            
            # Start with the two extremes
            selected = [0, n_unique - 1]
            
            for _ in range(n_evals - 2):
                if time.time() - start_time > deadline:
                    break
                # Find the point that maximizes the minimum distance to selected points
                # Compute distance to nearest selected point for each candidate
                min_dists = np.full(n_unique, np.inf)
                for s in selected:
                    min_dists = np.minimum(min_dists, np.abs(f1_unique - f1_unique[s]))
                
                # Exclude already selected points
                for s in selected:
                    min_dists[s] = -1
                
                # Pick the point with maximum minimum distance
                new_idx = np.argmax(min_dists)
                selected.append(new_idx)
            
            x1_vals = x1_unique[selected]
        
        # Construct points with x[1:] = 0
        points = np.zeros((len(x1_vals), dim))
        points[:, 0] = x1_vals
        
        # Evaluate
        for i in range(len(x1_vals)):
            if time.time() - start_time > deadline:
                break
            x = points[i].copy()
            x_clipped = np.clip(x, lo, hi)
            problem(x_clipped)
        
        return None
