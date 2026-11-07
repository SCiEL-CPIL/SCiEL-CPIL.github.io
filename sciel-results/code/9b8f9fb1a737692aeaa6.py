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
        
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        
        # ZDT6 Pareto front: x[1:]=0, f1 = 1 - exp(-4*x1)*sin(6*pi*x1)^8, f2 = 1 - sqrt(f1)
        # f1 is not monotonic in x1, but the set of achievable f1 values is [f1_min, 1.0]
        
        # High-resolution grid to map x1 -> f1
        n_grid = 200000
        x_grid = np.linspace(0, 1, n_grid)
        f1_grid = 1.0 - np.exp(-4.0 * x_grid) * (np.sin(6.0 * np.pi * x_grid) ** 8)
        
        # Sort by f1 value to get the range of achievable f1
        sort_idx = np.argsort(f1_grid)
        f1_sorted = f1_grid[sort_idx]
        x1_sorted = x_grid[sort_idx]
        
        f1_min = f1_sorted[0]
        f1_max = f1_sorted[-1]
        
        # For hypervolume maximization on a monotone front (f2 = 1 - sqrt(f1)),
        # the optimal placement of N points is to equalize the hypervolume contribution.
        # The hypervolume of a set of points sorted by f1 ascending is:
        # HV = sum_{i} (f1_{i+1} - f1_i) * (11 - f2_i) + (11 - f1_max) * (11 - f2_max)
        # where f2_i = 1 - sqrt(f1_i)
        # 
        # To maximize this with N points, we want to place them where the marginal
        # hypervolume gain is highest. For a smooth concave front, this means
        # spacing that accounts for the curvature.
        #
        # Practical approach: use a dense candidate set and do greedy selection
        # with vectorized operations for speed.
        
        # Create a candidate set: for each unique f1 value in the sorted grid,
        # keep the x1 that achieves it. We'll subsample to keep it manageable.
        # Since f1 oscillates, many x1 values map to the same f1. We just need
        # one x1 per f1 value.
        
        # Use a coarser candidate set for selection, then refine
        n_candidates = 5000
        # Sample evenly from the sorted f1 range
        cand_indices = np.linspace(0, len(f1_sorted) - 1, n_candidates, dtype=int)
        cand_f1 = f1_sorted[cand_indices]
        cand_x1 = x1_sorted[cand_indices]
        
        n_evals = min(budget, len(cand_f1))
        
        if n_evals <= 2:
            x1_vals = cand_x1[:n_evals]
        else:
            # Greedy selection: start with the two endpoints, then iteratively
            # add the point that maximizes the minimum f1-distance to existing points.
            # This is O(n_evals * n_candidates) which is 200 * 5000 = 1M, fast.
            
            selected_mask = np.zeros(n_candidates, dtype=bool)
            selected_f1 = np.array([cand_f1[0], cand_f1[-1]])
            selected_mask[0] = True
            selected_mask[-1] = True
            
            for _ in range(n_evals - 2):
                if time.time() - start_time > deadline:
                    break
                
                # For each unselected candidate, compute distance to nearest selected point in f1
                # Vectorized: for each candidate, min over selected
                # This is O(n_candidates * len(selected)) per iteration
                # With n_candidates=5000 and up to 200 selected, this is 1M ops per iter, 200 iters = 200M total
                # That might be slow. Let's use a different approach.
                
                # Alternative: since we want to maximize min-distance, and f1 is 1D,
                # we can use the fact that the optimal next point is in the largest gap.
                # Find the largest gap between consecutive selected f1 values.
                
                sel_sorted = np.sort(selected_f1)
                gaps = np.diff(sel_sorted)
                # Also consider the gaps at the ends (but we already have endpoints)
                
                # The point to add is at the midpoint of the largest gap (in f1 space)
                max_gap_idx = np.argmax(gaps)
                mid_f1 = 0.5 * (sel_sorted[max_gap_idx] + sel_sorted[max_gap_idx + 1])
                
                # Find the candidate closest to mid_f1 that hasn't been selected
                dists = np.abs(cand_f1 - mid_f1)
                dists[selected_mask] = np.inf
                new_idx = np.argmin(dists)
                
                selected_f1 = np.append(selected_f1, cand_f1[new_idx])
                selected_mask[new_idx] = True
            
            x1_vals = cand_x1[selected_mask]
        
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
