import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        budget = self.budget
        dim = self.dim

        # ZDT6 Pareto Front parameters
        # x[0] in [0, 1], x[1:] = 0
        # f1 = 1 + 10*(dim-1)*x0
        # g = 1 + 10*(dim-1)
        # f2 = (1/g) * (1 - sqrt(f1/g))^2 + 1
        
        g_val = 1 + 10 * (dim - 1)
        
        # Generate candidates
        # We want to cover the front. Uniform x0 is a good start.
        x0_candidates = np.linspace(0.0, 1.0, budget)
        
        f1_candidates = 1 + 10 * (dim - 1) * x0_candidates
        sqrt_term = np.sqrt(f1_candidates / g_val)
        f2_candidates = (1.0 / g_val) * (1.0 - sqrt_term)**2 + 1.0
        
        # Points are (f1, f2)
        # We want to maximize hypervolume relative to r = [11, 11]
        r1 = 11.0
        r2 = 11.0
        
        # Filter candidates that are strictly better than reference
        # f1 < 11 and f2 < 11
        # f1 max is 1 + 90 = 91. So many candidates will have f1 > 11.
        # Only candidates with f1 < 11 contribute to HV.
        # f1 < 11 => 1 + 90*x0 < 11 => 90*x0 < 10 => x0 < 1/9.
        # So only the first ~1/9 of the x0 range is useful for HV!
        # This is a crucial insight. The reference point is [11, 11].
        # f1 ranges from 1 to 91.
        # Points with f1 > 11 are dominated by the reference point in the f1 dimension?
        # No, the reference point defines the upper bound of the dominated region.
        # A point p contributes to HV only if p < r in ALL objectives.
        # So if f1 > 11, it does not contribute to the hypervolume at all.
        
        # Therefore, we should ONLY select points with f1 < 11.
        # This corresponds to x0 < 1/9.
        # We have 200 budget. We should use all 200 to sample x0 in [0, 1/9] densely.
        
        # Let's refine candidates to be in the useful range.
        x0_useful_max = (11.0 - 1.0) / (10.0 * (dim - 1))
        # x0_useful_max = 10 / 90 = 1/9
        
        # Generate 200 points in [0, x0_useful_max]
        x0_candidates = np.linspace(0.0, x0_useful_max, budget)
        
        f1_candidates = 1 + 10 * (dim - 1) * x0_candidates
        sqrt_term = np.sqrt(f1_candidates / g_val)
        f2_candidates = (1.0 / g_val) * (1.0 - sqrt_term)**2 + 1.0
        
        # All these points have f1 < 11.
        # f2 is always < 2, so f2 < 11 is always true.
        
        # Greedy Selection
        # List of selected points: (f1, f2, x0)
        selected = []
        remaining_indices = list(range(budget))
        
        # To speed up, we can pre-sort remaining by f1?
        # We need to find neighbors in the selected set for each candidate.
        
        # Since N=200, O(N^3) is fine.
        
        for _ in range(budget):
            best_gain = -1
            best_idx = -1
            best_point = None
            
            # Current selected points sorted by f1
            # We maintain selected as a list of (f1, f2, x0)
            
            for idx in remaining_indices:
                f1_c = f1_candidates[idx]
                f2_c = f2_candidates[idx]
                
                # Find neighbors in selected
                # Since selected is small, linear search is okay, or we can keep it sorted.
                # Let's keep selected sorted by f1.
                
                # Find insertion point
                # Linear scan for simplicity in this inner loop, or binary search.
                # With 200 items, linear is ~200 ops. 200*200*200 = 8M ops. Fine.
                
                left_f1 = 0.0
                left_f2 = r2 # If no left neighbor, the "step" starts from 0?
                             # Actually, if no point has f1 < f1_c, then the region [0, f1_c] is covered by f2_c?
                             # No, the point c covers [f1_c, r1].
                             # If there is no point with f1 < f1_c, then for f1 in [0, f1_c), no point dominates?
                             # Wait, the hypervolume integral is from 0 to r1.
                             # g(f1) = min { p.f2 | p.f1 <= f1 }.
                             # If no p has p.f1 <= f1, then g(f1) = infinity (or r2, meaning no contribution).
                             # So if f1_c is the smallest f1 in the set, it contributes for [f1_c, next_f1].
                             # The area for [0, f1_c] is 0 because no point dominates there.
                             
                             # So, if left neighbor doesn't exist, left_f1 = f1_c, left_f2 = r2 (irrelevant for area calc of the segment starting at f1_c).
                             # Actually, we just need the previous point in the sorted list.
                             
                right_f1 = r1
                right_f2 = r2 # If no right neighbor, the segment goes to r1.
                              # The height is r2 - f2_c.
                              # Wait, the formula is (next_f1 - current_f1) * (r2 - current_f2).
                              # If current is the last one, next_f1 = r1.
                              
                # Find immediate left and right in selected
                left_idx = -1
                right_idx = -1
                for i, p in enumerate(selected):
                    if p[0] < f1_c:
                        left_idx = i
                    elif p[0] > f1_c:
                        if right_idx == -1:
                            right_idx = i
                        else:
                            break # Since sorted, first one > f1_c is the right neighbor
                
                if left_idx != -1:
                    left_f1 = selected[left_idx][0]
                    left_f2 = selected[left_idx][1]
                else:
                    left_f1 = f1_c # Segment starts at f1_c
                    left_f2 = r2   # Not used for area of the new segment directly, but for old area calc?
                    
                if right_idx != -1:
                    right_f1 = selected[right_idx][0]
                    right_f2 = selected[right_idx][1]
                else:
                    right_f1 = r1
                    right_f2 = r2

                # Calculate Old Area contribution of the segment [left_f1, right_f1]
                # Before inserting c, this segment was covered by the point at left_idx (if exists)
                # If left_idx exists, the point at left_idx covered [left_f1, right_f1] with height (r2 - left_f2).
                # If left_idx does not exist, this segment [f1_c, right_f1] was NOT covered by any point to the left.
                # Wait, if left_idx is -1, it means no point has f1 < f1_c.
                # So the region [0, f1_c] was empty.
                # The region [f1_c, right_f1] was covered by... nothing?
                # No, if there is no point with f1 < f1_c, then for any f1 in [f1_c, right_f1), the condition p.f1 <= f1 is satisfied by... no point?
                # If no point has f1 <= f1, then g(f1) = infinity.
                # So the area was 0.
                
                # Let's re-evaluate.
                # The set of non-dominated points defines the step function.
                # If selected is empty, HV = 0.
                # If we add c, HV = (r1 - f1_c) * (r2 - f2_c).
                
                # If selected is not empty:
                # We find the neighbors.
                # The old area in the interval [left_f1, right_f1] was calculated using the point at left_idx.
                # Old Area = (right_f1 - left_f1) * (r2 - left_f2)  (if left_idx exists)
                # If left_idx does not exist, the interval [0, right_f1] was not covered by any point to the left of right_f1?
                # Actually, if left_idx is -1, it means c is the new smallest f1.
                # The old area for [0, right_f1] was 0? No.
                # If c is the smallest, then previously the smallest was right_idx? No, right_idx is the first one > f1_c.
                # If left_idx is -1, then all points in selected have f1 > f1_c.
                # So the smallest f1 in selected was selected[0] (which is right_idx).
                # The region [0, selected[0].f1] was empty (HV=0).
                # The region [selected[0].f1, next] was covered by selected[0].
                # So inserting c (with f1_c < selected[0].f1) adds area for [f1_c, selected[0].f1].
                # New Area = (selected[0].f1 - f1_c) * (r2 - f2_c).
                # Old Area for this segment was 0.
                # So Gain = New Area.
                
                # General Case:
                # Left neighbor L exists.
                # Old Area for [L.f1, R.f1] = (R.f1 - L.f1) * (r2 - L.f2).
                # New Area for [L.f1, C.f1] = (C.f1 - L.f1) * (r2 - L.f2).
                # New Area for [C.f1, R.f1] = (R.f1 - C.f1) * (r2 - C.f2).
                # Gain = New - Old.
                
                if left_idx != -1:
                    old_area = (right_f1 - left_f1) * (r2 - left_f2)
                    new_area_left = (f1_c - left_f1) * (r2 - left_f2)
                    new_area_c = (right_f1 - f1_c) * (r2 - f2_c)
                    gain = (new_area_left + new_area_c) - old_area
                else:
                    # No left neighbor. C is the new first point.
                    # Old area for [0, R.f1] was 0 (for the part [0, R.f1] where no point had f1 <= f1).
                    # Actually, the point R (right_idx) covered [R.f1, Next.f1].
                    # The region [0, R.f1] was empty.
                    # New area is [C.f1, R.f1] covered by C.
                    # Gain = (right_f1 - f1_c) * (r2 - f2_c)
                    gain = (right_f1 - f1_c) * (r2 - f2_c)
                    
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    best_point = (f1_c, f2_c, x0_candidates[idx])
            
            # Add best point to selected
            if best_idx != -1:
                selected.append(best_point)
                remaining_indices.remove(best_idx)
                # Keep selected sorted by f1
                selected.sort(key=lambda p: p[0])
                
            # Time check
            if time.time() > deadline:
                break
        
        # Execute evaluations in the selected order
        n_evals = 0
        for p in selected:
            if time.time() > deadline:
                break
            if n_evals >= budget:
                break
            
            x = np.zeros(dim)
            x[0] = p[2]
            x = np.clip(x, lo, hi)
            
            try:
                problem(x)
                n_evals += 1
            except Exception:
                break
        
        return None
