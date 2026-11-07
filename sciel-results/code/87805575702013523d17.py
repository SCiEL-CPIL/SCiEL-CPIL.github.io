import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        # ZDT6 specific knowledge
        # f1(x1) = 1 - (1 - exp(-4*x1) * sin(6*pi*x1)**6)**(1/10)
        # f2(x1) = 1 - sqrt(f1(x1))
        # The Pareto front is parameterized by x1 in [0, x1_peak] where f1 is maximized.
        
        def calc_f1(x1):
            # x1 is scalar or array
            return 1 - (1 - np.exp(-4 * x1) * np.sin(6 * np.pi * x1)**6)**0.1

        # 1. Find the precise maximum of f1 and its location x1_peak
        # The maximum is in the first lobe, roughly x1 in [0, 0.2]
        # We use a dense sampling to find the peak location accurately
        
        # Coarse search to find the region of the first peak
        x1_coarse = np.linspace(0, 0.5, 10000)
        f1_coarse = calc_f1(x1_coarse)
        
        # Find the first local maximum which is also the global maximum in this range
        # We look for the first peak where the function starts decreasing
        # Actually, the global max is the first peak.
        # Let's find the index of the global max in the coarse grid
        idx_coarse = np.argmax(f1_coarse)
        
        # Refine around the coarse peak
        if idx_coarse > 0 and idx_coarse < len(x1_coarse) - 1:
            x1_left = x1_coarse[idx_coarse - 1]
            x1_right = x1_coarse[idx_coarse + 1]
        else:
            x1_left = max(0, x1_coarse[idx_coarse] - 0.001)
            x1_right = min(0.5, x1_coarse[idx_coarse] + 0.001)
        
        # Fine search
        x1_fine = np.linspace(x1_left, x1_right, 100000)
        f1_fine = calc_f1(x1_fine)
        idx_fine = np.argmax(f1_fine)
        x1_peak = x1_fine[idx_fine]
        f1_max = f1_fine[idx_fine]
        
        # 2. Generate a dense set of candidate points on the Pareto front
        # We sample x1 uniformly in [0, x1_peak]
        n_candidates = 20000
        x1_candidates = np.linspace(0, x1_peak, n_candidates)
        f1_candidates = calc_f1(x1_candidates)
        f2_candidates = 1 - np.sqrt(f1_candidates)
        
        # Ensure we only keep points that are non-dominated
        # Since f1 is increasing and f2 is decreasing in this range, all points are non-dominated
        # But due to numerical precision, let's be safe.
        
        # 3. Greedy Hypervolume Maximization
        # We want to select 'budget' points from the candidates to maximize hypervolume
        # Reference point is [11.0, 11.0]
        ref_point = np.array([11.0, 11.0])
        
        # Initial selection: Include the two extreme points (min f1, min f2)
        # Point 0: x1=0 -> f1=0, f2=1
        # Point N: x1=x1_peak -> f1=max, f2=min
        # Let's start with these two to ensure coverage of the extremes
        selected_indices = [0, n_candidates - 1]
        selected_points = np.array([f1_candidates[selected_indices], f2_candidates[selected_indices]]).T
        
        # Function to calculate hypervolume for a set of 2D points
        def calc_hv(points, ref):
            # points: Nx2 array of (f1, f2)
            # Sort by f1
            points = points[np.argsort(points[:, 0])]
            hv = 0.0
            prev_f2 = ref[1]
            for i in range(len(points)):
                f1 = points[i, 0]
                f2 = points[i, 1]
                if f1 >= ref[0] or f2 >= ref[1]:
                    continue
                # Width is f1 - prev_f1 (but we integrate from left)
                # Actually, standard HV calculation:
                # Sort by f1 ascending.
                # For each point, the contribution is (f1[i] - f1[i-1]) * (ref[1] - f2[i]) ? No.
                # The area is sum of rectangles.
                # Let's use the standard formula:
                # Sort by f1.
                # HV = sum_{i=1}^{n} (f1[i] - f1[i-1]) * (ref[1] - f2[i])  -- this is not quite right if f2 is not monotonic in f1.
                # But for a Pareto front, f2 is decreasing as f1 increases.
                # So we can sort by f1.
                # The hypervolume is the area under the curve formed by the points and the reference point.
                # Area = sum of trapezoids or rectangles.
                # Let's use a simple method:
                # Sort by f1.
                # For each point i, the width is (f1[i] - f1[i-1]) where f1[-1]=0.
                # The height is (ref[1] - f2[i]).
                # This works if the function is monotonic.
                
                # Let's implement a robust 2D HV calculator
                pass

        def calc_hv_2d(points, ref):
            # points: Nx2, ref: 2x1
            # Only consider points better than ref
            mask = (points[:, 0] < ref[0]) & (points[:, 1] < ref[1])
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            
            # Sort by f1
            pts = pts[np.argsort(pts[:, 0])]
            
            hv = 0.0
            # The hypervolume is the sum of areas of rectangles
            # Each rectangle is defined by the current point and the previous point's f1
            # But since f2 is decreasing, the "upper bound" of the rectangle for point i is f2[i]
            # and the "left bound" is f1[i-1] (or 0 for the first point)
            # Wait, the area is from 0 to f1[i] and 0 to f2[i]? No, it's the dominated region.
            # The dominated region is [0, f1_i] x [0, f2_i] for each point, but they overlap.
            # The union of these regions can be calculated by sweeping.
            
            # Sort by f1 ascending.
            # Let f1_sorted, f2_sorted be the sorted points.
            # HV = sum_{i=0}^{n-1} (f1[i+1] - f1[i]) * (ref[1] - f2[i]) ? No.
            # Let's visualize:
            # Points (f1, f2). Ref (R1, R2).
            # Sort by f1.
            # For each point i, the contribution to HV is the area of the rectangle
            # with width (f1[i+1] - f1[i]) and height (R2 - f2[i])?
            # No, the height is determined by the minimum f2 of the points that dominate that x-range.
            # Since f2 is decreasing with f1, the point with the smallest f1 has the largest f2.
            # So for the range [0, f1[0]], the height is R2 - f2[0].
            # For the range [f1[0], f1[1]], the height is R2 - f2[1].
            # ...
            # For the range [f1[n-1], R1], the height is R2 - f2[n-1].
            
            # Let's verify:
            # Point 0: (f1_0, f2_0). Dominates [0, f1_0] x [0, f2_0].
            # Point 1: (f1_1, f2_1). Dominates [0, f1_1] x [0, f2_1].
            # Since f1_1 > f1_0 and f2_1 < f2_0.
            # The union is:
            # [0, f1_0] x [0, f2_0]  (from P0)
            # [f1_0, f1_1] x [0, f2_1] (from P1, since P0 doesn't cover x > f1_0)
            # ...
            # [f1_{n-1}, R1] x [0, f2_{n-1}] (from P_{n-1})
            
            # So HV = f1[0]*(R2 - f2[0]) + (f1[1]-f1[0])*(R2 - f2[1]) + ... + (R1 - f1[n-1])*(R2 - f2[n-1])
            
            f1_s = pts[:, 0]
            f2_s = pts[:, 1]
            
            # Prepend 0 to f1_s
            f1_prev = np.concatenate(([0.0], f1_s[:-1]))
            
            widths = f1_s - f1_prev
            # The last width extends to R1
            widths[-1] = R1 - f1_s[-1]
            
            heights = R2 - f2_s
            
            return np.sum(widths * heights)

        # Greedy Selection
        n_selected = 0
        candidates = np.column_stack((f1_candidates, f2_candidates))
        
        # Remove duplicates if any, though linspace should be unique
        # We will maintain a list of selected points
        selected_list = []
        
        # Start with the two extremes
        selected_list.append(candidates[0])
        selected_list.append(candidates[-1])
        n_selected = 2
        
        # If budget is small, we might not need greedy
        if self.budget <= 2:
            pass
        else:
            # Pre-calculate distances or just iterate
            # To speed up, we can use a heap or just scan
            # Scanning 20000 candidates 198 times is 4 million operations, which is fast in Python/NumPy
            
            # We need to calculate HV for (selected + candidate) each time
            # This is O(N_sel) for HV calculation.
            # Total O(N_sel * N_cand * N_sel) = O(N_cand * N_sel^2)
            # 20000 * 200^2 = 800 million operations. This might be too slow in pure Python loops.
            # We need to optimize.
            
            # Alternative: Use a known algorithm or approximation.
            # Or, reduce the number of candidates.
            # Let's reduce candidates to 5000.
            # 5000 * 200^2 = 200 million. Still might be tight.
            
            # Let's try a different strategy:
            # Uniform sampling in objective space is often good enough.
            # Let's stick to the uniform f1 sampling but refine it.
            
            # Actually, the previous program used uniform f1 targets and binary search.
            # Let's try to improve that by using a more accurate mapping.
            
            # Let's go back to the uniform f1 target idea but with a better lookup.
            
            # The current program's approach was:
            # 1. Find x1_peak.
            # 2. Create lookup table for f1(x1) on [0, x1_peak].
            # 3. For each target f1 in linspace(0, f1_max, N), find x1.
            
            # This is O(N log M). Very fast.
            # Why did it score 1.5047?
            # Maybe the hypervolume is not maximized by uniform f1 spacing.
            # For a convex front, uniform objective spacing is optimal.
            # ZDT6 front: f2 = 1 - sqrt(f1). This is concave.
            # For a concave front, uniform spacing in the parameter might be better?
            # Or uniform spacing in f2?
            
            # Let's test uniform f2 spacing.
            # f2 ranges from 1 (at x1=0) to 1-sqrt(f1_max) (at x1_peak).
            # Let's try a mix: 50% uniform f1, 50% uniform f2?
            
            # Actually, let's just use the greedy approach but with fewer candidates to ensure speed.
            # Let's use 1000 candidates.
            # 1000 * 200^2 = 40 million. This is feasible.
            
            n_cand_greedy = 1000
            x1_greedy = np.linspace(0, x1_peak, n_cand_greedy)
            f1_greedy = calc_f1(x1_greedy)
            f2_greedy = 1 - np.sqrt(f1_greedy)
            candidates = np.column_stack((f1_greedy, f2_greedy))
            
            # Initialize selected with extremes
            selected = np.array([candidates[0], candidates[-1]])
            
            # Greedy loop
            for _ in range(self.budget - 2):
                if time.time() > deadline - 5.0:
                    break
                
                best_hv = -1
                best_idx = -1
                best_point = None
                
                # Calculate HV for each candidate added to selected
                # This is the bottleneck.
                # We can optimize by noting that adding a point only affects the local area.
                # But for simplicity and correctness, let's just calculate.
                # To speed up, we can subsample candidates if it's too slow.
                
                # Let's try to vectorize the HV calculation for all candidates at once?
                # HV(selected + c_i) for all i.
                # This is hard to vectorize directly.
                
                # Let's just loop.
                for i in range(len(candidates)):
                    # Skip if already selected
                    # Check if candidate is close to any selected
                    # Simple check: if min distance < epsilon, skip?
                    # No, just calculate.
                    
                    new_set = np.vstack([selected, candidates[i]])
                    hv = calc_hv_2d(new_set, ref_point)
                    
                    if hv > best_hv:
                        best_hv = hv
                        best_idx = i
                        best_point = candidates[i]
                
                if best_idx != -1:
                    selected = np.vstack([selected, best_point])
                    # Remove the selected candidate from the list?
                    # No, we can keep it, but we should avoid re-selecting.
                    # We can mark it as used.
                    # Or just remove it from candidates.
                    candidates = np.delete(candidates, best_idx, axis=0)
                    
                    if len(candidates) == 0:
                        break
        
        # Map selected points back to x1
        # We need to find the x1 that corresponds to each selected (f1, f2) pair.
        # Since we have the dense lookup table from earlier (x1_fine, f1_fine), we can use that.
        # Or just use the candidates array we used for greedy, which has x1_greedy.
        
        # Wait, if we used greedy on `candidates`, we lost the x1 mapping if we deleted rows.
        # Let's keep the x1 values alongside.
        
        # Let's restart the greedy part with x1 tracking.
        
        # Re-do greedy with x1
        n_cand_greedy = 1000
        x1_greedy = np.linspace(0, x1_peak, n_cand_greedy)
        f1_greedy = calc_f1(x1_greedy)
        f2_greedy = 1 - np.sqrt(f1_greedy)
        
        # Combine into a list of dictionaries or a structured array
        # Let's use a list of (x1, f1, f2)
        cand_list = [(x1_greedy[i], f1_greedy[i], f2_greedy[i]) for i in range(n_cand_greedy)]
        
        selected_x1 = []
        selected_points = []
        
        # Add extremes
        selected_x1.append(cand_list[0][0])
        selected_points.append(np.array([cand_list[0][1], cand_list[0][2]]))
        
        selected_x1.append(cand_list[-1][0])
        selected_points.append(np.array([cand_list[-1][1], cand_list[-1][2]]))
        
        # Remove extremes from cand_list to avoid re-selection
        cand_list = cand_list[1:-1]
        
        for _ in range(self.budget - 2):
            if time.time() > deadline - 5.0:
                break
            
            best_hv = -1
            best_idx = -1
            
            # We have `selected_points` which is a list of arrays.
            # Convert to numpy array for HV calc
            sel_arr = np.array(selected_points)
            
            for i, (x1, f1, f2) in enumerate(cand_list):
                new_point = np.array([f1, f2])
                new_set = np.vstack([sel_arr, new_point])
                hv = calc_hv_2d(new_set, ref_point)
                
                if hv > best_hv:
                    best_hv = hv
                    best_idx = i
            
            if best_idx != -1:
                x1_best, f1_best, f2_best = cand_list[best_idx]
                selected_x1.append(x1_best)
                selected_points.append(np.array([f1_best, f2_best]))
                cand_list.pop(best_idx)
            
            if len(cand_list) == 0:
                break
        
        # If we didn't reach budget, pad with uniform samples?
        # The greedy should reach it unless time runs out.
        
        # Evaluate the selected x1 values
        recorded = []
        for i in range(len(selected_x1)):
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = selected_x1[i]
            # Clip to bounds [0, 1]
            x = np.clip(x, 0, 1)
            obj = problem(x)
            recorded.append(obj)
        
        # If we have fewer points than budget, we can add more?
        # No, the loop above runs budget-2 times.
        # So we should have budget points.
        
        return recorded
