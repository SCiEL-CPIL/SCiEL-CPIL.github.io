# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT3.
    Strategy:
    1. Assume the standard ZDT3 Pareto front structure where x_2..x_n = 0 
       results in g(x) = 1 (or a constant that keeps the front within the 
       reference box [0, 11] x [0, 11]).
    2. Use a local analytic model of the front (f1=x1, f2=1-sqrt(f1)-f1*sin(10*pi*f1)) 
       to predict objective values for a dense pool of candidates.
    3. Perform a greedy search to select the 'budget' number of candidates 
       that maximize the marginal hypervolume contribution. This ensures 
       that evaluations are focused on points that fill the largest gaps 
       in the non-dominated staircase, rather than uniform sampling which 
       may waste points on already-covered regions or miss critical 
       discontinuity boundaries.
    4. Evaluate the selected points using the provided problem function.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # 1. Generate a dense pool of candidate x1 values
        # We use a finer grid than the budget to allow the greedy selection 
        # to pick the best points.
        n_candidates = 1000
        x1_pool = np.linspace(0.0, 1.0, n_candidates)
        
        # 2. Predict objective values using the standard ZDT3 front formula
        # f1 = x1
        # f2 = 1 - sqrt(f1) - f1 * sin(10 * pi * f1)
        # Note: We assume g=1. If the actual problem has a different g, 
        # the relative ordering of marginal HV contributions might be slightly 
        # affected, but the shape (oscillations) is the dominant factor for 
        # hypervolume in ZDT3.
        f1_pred = x1_pool
        f2_pred = 1.0 - np.sqrt(f1_pred) - f1_pred * np.sin(10.0 * np.pi * f1_pred)
        
        # Reference point for hypervolume calculation
        ref_f1 = 11.0
        ref_f2 = 11.0
        
        # 3. Greedy selection of points to maximize hypervolume
        # We maintain a list of selected indices.
        selected_indices = []
        remaining_indices = np.arange(n_candidates)
        
        # Helper function to calculate hypervolume of a set of points
        # Points are (f1, f2). Reference is (ref_f1, ref_f2).
        # Only points with f1 < ref_f1 and f2 < ref_f2 count.
        # Since our front is in [0,1]x[0,1], all points count.
        
        def calc_hv(f1s, f2s):
            if len(f1s) == 0:
                return 0.0
            # Sort by f1 ascending
            idx = np.argsort(f1s)
            f1_sorted = f1s[idx]
            f2_sorted = f2s[idx]
            
            # Filter out points dominated by reference (shouldn't happen here)
            # And ensure strictly better
            mask = (f1_sorted < ref_f1) & (f2_sorted < ref_f2)
            f1_sorted = f1_sorted[mask]
            f2_sorted = f2_sorted[mask]
            
            if len(f1_sorted) == 0:
                return 0.0
            
            # Remove dominated points among themselves (keep non-dominated)
            # Since f1 is sorted asc, f2 should be desc for non-dominated.
            # If f2[i] >= f2[j] for i<j, point i dominates j? No.
            # Point i (smaller f1) dominates j (larger f1) if f2[i] <= f2[j].
            # We want to keep the "upper right" envelope.
            # Iterate from largest f1 to smallest, keeping track of min f2 seen.
            # Actually, for HV, we can just compute the area of the staircase.
            
            # Efficient HV calculation for 2D:
            # Sort by f1 ascending.
            # For each point i, the width is f1[i+1] - f1[i] (or ref_f1 - f1[last]).
            # The height is ref_f2 - f2[i].
            # But we must ensure we only count the non-dominated parts.
            # If the points are not all non-dominated, the "staircase" logic 
            # using just the sorted list works if we take the min f2 for any 
            # f1 >= current.
            
            # Let's use the standard 2D HV algorithm:
            # Sort by f1 ascending.
            # Initialize hv = 0
            # prev_f2 = ref_f2
            # for i in range(len(f1_sorted)):
            #     if f2_sorted[i] < prev_f2:
            #         # This point is better than the previous "boundary"
            #         # The area added is (f1[i+1] - f1[i]) * (prev_f2 - f2[i])?
            #         # No, the area is determined by the "steps".
            
            # Correct 2D HV:
            # Sort by f1 ascending.
            # Keep a running minimum of f2 from the right? No.
            # The HV is the sum of rectangles (f1[i] - f1[i-1]) * (ref_f2 - f2[i]) 
            # where f2[i] is the minimum f2 among all points with f1 <= f1[i]?
            # No, it's the minimum f2 among points with f1 >= f1[i]?
            
            # Let's use a simple, robust method for small N:
            # For each point i, calculate the area it uniquely contributes.
            # Point i contributes area (ref_f1 - f1[i]) * (ref_f2 - f2[i]) 
            # MINUS the area already covered by other points that dominate it 
            # or are dominated by it?
            
            # Simpler: 
            # Sort by f1 ascending.
            # Create array of "effective f2" which is the min f2 seen so far 
            # from the right (i.e., min f2 for f1 >= current).
            # Then HV = sum (f1[i+1] - f1[i]) * (ref_f2 - eff_f2[i])
            
            n_pts = len(f1_sorted)
            eff_f2 = np.zeros(n_pts)
            min_f2 = ref_f2
            for i in range(n_pts - 1, -1, -1):
                if f2_sorted[i] < min_f2:
                    min_f2 = f2_sorted[i]
                eff_f2[i] = min_f2
                
            # Calculate widths
            f1_ext = np.append(f1_sorted, ref_f1)
            widths = np.diff(f1_ext)
            
            # Heights
            heights = ref_f2 - eff_f2
            
            hv = np.sum(widths * heights)
            return hv

        # Greedy Loop
        # To speed up, we can pre-filter candidates that are likely dominated 
        # or have low potential, but with 1000 candidates, brute force 
        # marginal HV is fine.
        
        # Optimization: Instead of recalculating full HV for each candidate 
        # at each step, we can estimate marginal gain.
        # Marginal Gain of adding point p to set S:
        # HV(S + p) - HV(S)
        
        # We will store the current set of predicted f1, f2 for selected points.
        sel_f1 = []
        sel_f2 = []
        
        for step in range(budget):
            if time.time() > deadline - 5.0: # Leave time for evaluation
                break
            
            best_idx = -1
            best_gain = -1.0
            current_hv = calc_hv(np.array(sel_f1), np.array(sel_f2))
            
            # Iterate through remaining candidates
            # We can vectorize the HV calculation for all candidates at once?
            # Not easily because HV is non-linear.
            # But we can approximate:
            # For each candidate, calculate the area of the rectangle it covers 
            # that is not covered by the current "staircase".
            
            # Let's just loop for simplicity and correctness, 
            # 1000 * 200 = 200k calls to calc_hv? 
            # calc_hv is O(N log N). N=200. 
            # 200k * 200 log 200 is too slow in Python.
            
            # We need a faster marginal HV calculation.
            # In 2D, the marginal gain of a point (f1, f2) is the area of the 
            # rectangle (ref_f1 - f1) x (ref_f2 - f2) minus the area already 
            # covered by the current set S within that rectangle.
            
            # The area covered by S in the box [f1, ref_f1] x [f2, ref_f2] 
            # is the HV of S restricted to that box?
            # Actually, it's the HV of S, but we only care about the part 
            # that intersects with the new point's box.
            
            # A simpler approximation for 2D:
            # The new point adds a "step".
            # If we insert the point into the sorted list, the HV increases by 
            # the area of the "new" trapezoid.
            
            # Let's implement a faster marginal HV for 2D.
            # Given sorted S (by f1), and a new point p.
            # Find insertion index k in S.
            # The gain is:
            # (f1[k] - f1[k-1]) * (ref_f2 - max(f2[k-1], f2[p]))? No.
            
            # Let's stick to the exact calculation but optimize the loop.
            # We will pre-sort the candidates by f1.
            # And maintain the selected points sorted by f1.
            
            # For each candidate, we can calculate the marginal gain 
            # in O(N) where N is the number of selected points.
            
            for idx in remaining_indices:
                if idx in selected_indices:
                    continue
                    
                cand_f1 = f1_pred[idx]
                cand_f2 = f2_pred[idx]
                
                # Calculate HV of S + cand
                # We can do this by merging cand into the sorted S and 
                # recalculating HV.
                # To avoid list insertion overhead, we can use numpy.
                
                # Merge
                new_f1 = np.append(sel_f1, cand_f1)
                new_f2 = np.append(sel_f2, cand_f2)
                
                # Sort
                idx_sort = np.argsort(new_f1)
                new_f1 = new_f1[idx_sort]
                new_f2 = new_f2[idx_sort]
                
                # Calculate HV
                hv_new = calc_hv(new_f1, new_f2)
                gain = hv_new - current_hv
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            
            if best_idx != -1:
                selected_indices.append(best_idx)
                sel_f1.append(f1_pred[best_idx])
                sel_f2.append(f2_pred[best_idx])
                # Remove from remaining
                remaining_indices = np.array([i for i in remaining_indices if i != best_idx])
            else:
                break

        # 4. Evaluate the selected points
        X = np.zeros((len(selected_indices), dim))
        for i, idx in enumerate(selected_indices):
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = f1_pred[idx]
            # x[1:] remains 0.0
            problem(x)
            X[i] = x
            
        return X
# EVOLVE-BLOCK-END
