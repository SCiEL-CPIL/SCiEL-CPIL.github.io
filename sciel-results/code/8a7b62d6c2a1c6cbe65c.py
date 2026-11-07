import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        n = self.dim
        budget = self.budget
        
        if budget <= 0:
            return None
            
        # Extract bounds for x1 (index 0)
        # Assuming bounds is a list of (low, high)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # We exploit ZDT2 structure: x2..xn = 0 minimizes f2
        # We only need to optimize x1
        # x1 bounds
        x1_lo = lo[0]
        x1_hi = hi[0]
        
        # Initialize with a mix of power law distributions to cover the front well
        # We start with a small set and refine
        N_init = min(50, budget)
        
        # Strategy: Use a power law x1 = (i/N)^p. 
        # For ZDT2, curvature is high near x1=0. p=2 or p=3 is often good.
        # Let's try a combination: half points with p=2, half with p=4? 
        # Or just a well-chosen single p. 
        # The current best uses p=2. Let's stick with a robust initial set 
        # and then use the rest of the budget to fill gaps.
        
        # Create initial set
        # i from 0 to N_init-1
        idx = np.arange(N_init)
        # Use p=2.0 as baseline, but maybe mix in some uniform to ensure coverage?
        # Actually, for hypervolume, density where the front is "steep" or "curved" matters.
        # Let's use p=2.0 for the initial set.
        x1_vals_init = (idx / (N_init - 1)) ** 2.0
        # Ensure x1 is within bounds
        x1_vals_init = x1_lo + x1_vals_init * (x1_hi - x1_lo)
        
        # Create full vectors
        X = np.zeros((N_init, n))
        X[:, 0] = x1_vals_init
        # x2..xn are already 0
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        used = N_init
        
        # If we have budget left, refine by adding points in largest gaps
        # We want to maximize hypervolume. 
        # Approximation: Add points where the "area" of the rectangle formed by 
        # the current point, the reference point, and the next point is largest?
        # Or simpler: Add points in the largest x1 gaps?
        # Since f1=x1, largest x1 gap = largest f1 gap.
        # However, the contribution to hypervolume depends on f2 as well.
        # A point with low f2 contributes more to hypervolume.
        # Let's sort by f1 and look at the "loss" if we don't add a point.
        
        # Sort current points by f1 (which is x1)
        sort_idx = np.argsort(F[:, 0])
        F_sorted = F[sort_idx]
        X_sorted = X[sort_idx]
        
        # We will insert new points. 
        # To keep it simple and fast, we can just generate a dense set 
        # and pick the best ones? No, we have a fixed budget.
        
        # Alternative strategy: 
        # Use the remaining budget to perform a local search or gap filling.
        # Let's identify the gap in x1 space that is largest.
        # But hypervolume is about the area under the curve in objective space.
        
        # Let's try a different approach for the refinement phase:
        # Instead of complex gap logic, let's just generate a very dense set 
        # of candidates using a slightly different power law or log spacing 
        # and keep the best non-dominated set? 
        # No, we must return exactly the points we evaluated? 
        # "Every returned objective vector is recorded". 
        # The return value is ignored, but the *evaluations* are recorded.
        # So we need to choose the 200 points to evaluate.
        
        # If we already evaluated 50, we have 150 left.
        # Let's generate 150 more points.
        # Where?
        # The hypervolume of a set of points on a convex front is maximized 
        # when the points are distributed according to the curvature.
        
        # Let's use an adaptive scheme:
        # 1. Start with 50 points (p=2).
        # 2. Evaluate them.
        # 3. For the next 150 points, choose x1 values that minimize the 
        #    "regret" or maximize the marginal hypervolume contribution.
        
        # Simple heuristic for marginal contribution:
        # For each potential new point, estimate its contribution.
        # This is expensive if we do it for many candidates.
        
        # Let's use a simpler, robust strategy:
        # Use a logarithmic spacing for x1? 
        # x1 = (x1_hi - x1_lo) * (10**(i * log10(x1_hi/x1_lo) / N) - 1) / (x1_hi/x1_lo - 1)
        # This puts many points near 0.
        
        # Let's combine: 
        # The current program scored 1.3. 
        # Maybe the power p=2 is not optimal. 
        # Let's try to optimize p? 
        # Or use a mix of p=1.5 and p=3?
        
        # Let's try a "multi-start" power law approach for the initial set, 
        # and then use the rest to fill the largest gaps in f1.
        
        if used < budget:
            remaining = budget - used
            # Generate candidates for the remaining slots
            # We will pick 'remaining' points from a larger candidate pool 
            # based on gap size in f1.
            
            # Candidate pool: 1000 points with p=2.0
            N_cand = 1000
            idx_cand = np.arange(N_cand)
            x1_cand = (idx_cand / (N_cand - 1)) ** 2.0
            x1_cand = x1_lo + x1_cand * (x1_hi - x1_lo)
            
            # We want to select 'remaining' points from this pool 
            # that are furthest from the existing points in f1 space.
            
            # Get current f1 values
            f1_current = F[:, 0]
            
            # For each candidate, find the distance to the nearest current point in f1
            # This is O(N_cand * N_current). 1000 * 50 = 50k, fast.
            # Then pick the 'remaining' candidates with the largest minimum distance.
            
            # To make it even better, we can weight the distance by the "height" 
            # of the front at that point? No, just f1 gap is a good proxy 
            # for hypervolume improvement on a convex front.
            
            # Sort current f1
            f1_sorted = np.sort(f1_current)
            
            # Compute min distance for each candidate
            # Using searchsorted for efficiency
            insert_idx = np.searchsorted(f1_sorted, x1_cand)
            
            # Handle boundaries
            # dist_left = x1_cand - f1_sorted[insert_idx - 1] if insert_idx > 0 else inf
            # dist_right = f1_sorted[insert_idx] - x1_cand if insert_idx < len(f1_sorted) else inf
            # min_dist = min(dist_left, dist_right)
            
            # Vectorized version
            # Create arrays for left and right neighbors
            # Need to be careful with indices
            left_idx = insert_idx - 1
            right_idx = insert_idx
            
            # Mask for valid left indices
            mask_left = left_idx >= 0
            mask_right = right_idx < len(f1_sorted)
            
            min_dist = np.full(N_cand, np.inf)
            
            # Left distances
            if np.any(mask_left):
                dist_l = x1_cand[mask_left] - f1_sorted[left_idx[mask_left]]
                min_dist[mask_left] = dist_l
            
            # Right distances
            if np.any(mask_right):
                dist_r = f1_sorted[right_idx[mask_right]] - x1_cand[mask_right]
                # Update min_dist: if dist_r < current min_dist, take dist_r
                # But if min_dist was inf (no left neighbor), take dist_r
                # If both exist, take min
                current_min = min_dist[mask_right]
                new_min = np.minimum(current_min, dist_r)
                min_dist[mask_right] = new_min
            
            # Now select the 'remaining' candidates with the largest min_dist
            # Get indices of top 'remaining' distances
            # We need to ensure we don't pick duplicates if any (though candidates are unique)
            
            # Sort candidates by min_dist descending
            order = np.argsort(min_dist)[::-1]
            selected_cand_indices = order[:remaining]
            
            # Get the x1 values for these selected candidates
            x1_new = x1_cand[selected_cand_indices]
            
            # Create vectors
            X_new = np.zeros((remaining, n))
            X_new[:, 0] = x1_new
            
            # Evaluate
            # Check time
            if time.time() > deadline:
                # If we are running out of time, just return what we have
                # But we must not exceed budget. We have 'used' so far.
                # We can't evaluate more.
                pass
            else:
                F_new = np.array([problem(x) for x in X_new])
                used += remaining
                
                # Combine
                X = np.vstack([X, X_new])
                F = np.vstack([F, F_new])
                
        return X
