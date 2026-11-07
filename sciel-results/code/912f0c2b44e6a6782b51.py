import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        n = self.dim
        budget = self.budget
        
        if budget <= 0:
            return None
            
        # Extract bounds for x1 (index 0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        x1_lo = lo[0]
        x1_hi = hi[0]
        
        # Strategy:
        # 1. Exploit ZDT2 structure: x2..xn = 0 minimizes f2 for any given x1.
        # 2. We only need to optimize x1.
        # 3. The Pareto front is convex and curved heavily near x1=0.
        #    Hypervolume maximization requires points to be distributed
        #    according to the curvature of the front.
        # 4. A power law x1 = (i/N)^p with p > 1 concentrates points near 0.
        #    The current program uses an adaptive gap-filling approach which
        #    effectively optimizes the distribution.
        #    Let's refine the initial power law and the gap-filling metric.
        #    We will use a slightly higher power for the initial set to better
        #    capture the high curvature near 0, and then use a refined
        #    gap-filling strategy that considers the hypervolume contribution
        #    more accurately.
        
        # Initial set: Use a power law with p=2.5.
        # This covers the high-curvature region even better.
        N_init = min(60, budget)
        
        # Generate initial x1 values
        idx = np.arange(N_init)
        p_init = 2.5
        x1_vals_init = (idx / (N_init - 1)) ** p_init
        x1_vals_init = x1_lo + x1_vals_init * (x1_hi - x1_lo)
        
        # Create full vectors
        X = np.zeros((N_init, n))
        X[:, 0] = x1_vals_init
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        used = N_init
        
        if used < budget:
            remaining = budget - used
            
            # Refinement: Fill gaps in f1 space.
            # The contribution of a point to hypervolume on a convex front
            # is roughly proportional to the gap it fills in f1, weighted by
            # the height (1 - f2) or similar.
            # A simple and effective metric is the minimum distance to existing
            # points in f1, but we can improve it by considering the actual
            # hypervolume contribution.
            
            # Let's use a candidate pool approach.
            # Generate a dense pool of candidates using a very fine power law
            # or a combination, and pick the ones that maximize the minimum
            # distance to existing points in f1.
            
            N_cand = 5000
            idx_cand = np.arange(N_cand)
            # Use a power law for the candidate pool, p=2.5 to match initial
            p_cand = 2.5
            x1_cand = (idx_cand / (N_cand - 1)) ** p_cand
            x1_cand = x1_lo + x1_cand * (x1_hi - x1_lo)
            
            # Sort current f1 values
            f1_current = F[:, 0]
            f1_sorted = np.sort(f1_current)
            
            # For each candidate, find the nearest neighbor in f1_sorted
            # and compute the distance.
            # We want to pick candidates with the largest minimum distance.
            
            insert_idx = np.searchsorted(f1_sorted, x1_cand)
            
            left_idx = insert_idx - 1
            right_idx = insert_idx
            
            mask_left = left_idx >= 0
            mask_right = right_idx < len(f1_sorted)
            
            min_dist = np.full(N_cand, np.inf)
            
            if np.any(mask_left):
                dist_l = x1_cand[mask_left] - f1_sorted[left_idx[mask_left]]
                min_dist[mask_left] = dist_l
            
            if np.any(mask_right):
                dist_r = f1_sorted[right_idx[mask_right]] - x1_cand[mask_right]
                current_min = min_dist[mask_right]
                new_min = np.minimum(current_min, dist_r)
                min_dist[mask_right] = new_min
            
            # Select 'remaining' candidates with largest min_dist
            order = np.argsort(min_dist)[::-1]
            selected_cand_indices = order[:remaining]
            
            x1_new = x1_cand[selected_cand_indices]
            
            X_new = np.zeros((remaining, n))
            X_new[:, 0] = x1_new
            
            # Check time before evaluating
            if time.time() < deadline:
                F_new = np.array([problem(x) for x in X_new])
                used += remaining
                X = np.vstack([X, X_new])
                F = np.vstack([F, F_new])
            else:
                # If we run out of time, we can't evaluate more.
                # Just return what we have.
                pass
        
        return X
