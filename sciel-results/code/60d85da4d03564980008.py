import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        n = self.dim
        # ZDT6 specific structure:
        # f1 = x1
        # g = 10 - 18*x1 + sum_{i=2..n} (xi^2 - 2*xi*x1*cos(20*pi*xi*x1))
        # f2 = g * (1 - sqrt(f1/g))
        # The Pareto front is achieved when g is minimized for a given x1.
        # For x1 in [0, 1], the minimum of the sum term is achieved when xi=0 for all i>1.
        # Then g_min(x1) = 10 - 18*x1.
        # However, g must be positive for f2 to be real and defined in the standard way.
        # If g <= 0, the problem definition might clip or return inf/nan.
        # The reference point is [11, 11].
        
        # We want to maximize hypervolume of the set of points (f1, f2).
        # The hypervolume indicator for a set of non-dominated points sorted by f1
        # is the sum of rectangles.
        # Since we control the x1 values, we can compute the resulting (f1, f2) pairs
        # analytically and then optimize the choice of x1 values to maximize the
        # approximated hypervolume.
        
        # The Pareto front is a curve. We are choosing 200 points on this curve.
        # To maximize the area under the curve (with respect to the reference point),
        # we should distribute points such that the "steps" in the staircase approximation
        # are as small as possible in terms of area loss.
        
        # Let's discretize the possible x1 values into a fine grid (e.g., 1000 points)
        # and use dynamic programming to select the best 200 points from this grid
        # that maximize the hypervolume.
        
        # 1. Generate candidate points
        # We need to ensure g > 0.
        # g_min(x1) = 10 - 18*x1.
        # g_min(x1) > 0 => x1 < 10/18 = 5/9 ≈ 0.555.
        # If x1 > 5/9, g_min is negative.
        # The problem statement says "problem(x) clips... and returns 2 objective values".
        # Usually ZDT6 is defined for x1 in [0, 1].
        # If g is negative, sqrt(f1/g) is undefined.
        # However, the standard ZDT6 definition often assumes the domain where g>0 or handles it.
        # Let's check the behavior. If we pick x1 such that g_min < 0, and we set xi=0,
        # the evaluator might return NaN or Inf.
        # To be safe, we should only pick x1 such that g_min(x1) > 0.
        # So x1 in [0, 5/9].
        
        # Wait, is it possible to have g > 0 with x1 > 5/9 by choosing xi != 0?
        # g(x) = 10 - 18*x1 + sum(xi^2 - 2*xi*x1*cos(20*pi*xi*x1))
        # The term xi^2 - 2*xi*x1*cos(...) can be negative.
        # Minimum of xi^2 - 2*xi*x1*cos(theta) is -x1^2 when cos(theta)=1 and xi=x1.
        # So min g(x) for fixed x1 is 10 - 18*x1 - (n-1)*x1^2.
        # Let's check when this is positive.
        # 10 - 18*x1 - 9*x1^2 > 0 (for n=10)
        # Roots of 9*x1^2 + 18*x1 - 10 = 0
        # x1 = (-18 + sqrt(324 - 4*9*(-10))) / 18 = (-18 + sqrt(324+360)) / 18 = (-18 + sqrt(684)) / 18
        # sqrt(684) ≈ 26.15
        # x1 ≈ 8.15 / 18 ≈ 0.45.
        # So for x1 > 0.45, even the global minimum of g is negative.
        # Thus, we are restricted to x1 in [0, ~0.45] to have valid real-valued f2.
        
        # Let's verify the range.
        # If we restrict to x1 where g_min(x1) > 0 using xi=0 strategy:
        # x1 < 5/9 = 0.555.
        # But if we can adjust xi, we might push it further? No, xi=0 minimizes the sum term 
        # only if the cosine term is positive. 
        # Actually, for small xi, xi^2 - 2*xi*x1*cos(...) ≈ xi^2 - 2*xi*x1.
        # This is minimized at xi=x1 with value -x1^2.
        # So the true minimum of g for a given x1 is 10 - 18*x1 - (n-1)*x1^2.
        # We should sample x1 such that this minimum is positive to ensure valid points.
        
        # Let's define the analytic front points.
        # For a given x1, we want to minimize g.
        # We can try to find xi that minimizes g numerically for each x1? 
        # No, budget is 200 evaluations. We cannot do inner optimization.
        # We must assume a fixed mapping from x1 to x.
        # The standard "analytic front" assumption for ZDT6 is often xi=0.
        # If xi=0, g = 10 - 18*x1.
        # This is valid for x1 < 5/9.
        # f1 = x1.
        # f2 = (10 - 18*x1) * (1 - sqrt(x1 / (10 - 18*x1)))
        
        # Let's check if we can do better by setting xi to something else.
        # If we set xi such that cos(20*pi*xi*x1) = 1 and xi = x1, then term is -x1^2.
        # This reduces g, which increases f2 (since f2 = g * (1 - sqrt(f1/g))).
        # Wait, if g decreases, f2 increases?
        # Let h(g) = g * (1 - sqrt(f1/g)) = g - sqrt(f1*g).
        # dh/dg = 1 - sqrt(f1)/(2*sqrt(g)).
        # If g is small, dh/dg is negative. So decreasing g increases f2?
        # No, if g is small, sqrt(f1)/(2*sqrt(g)) is large, so derivative is negative.
        # So decreasing g INCREASES f2?
        # Let's check: f2 = g - sqrt(f1*g).
        # If g -> 0, f2 -> 0.
        # If g is large, f2 ~ g.
        # So smaller g gives smaller f2?
        # Let's re-evaluate.
        # f2 = g * (1 - sqrt(x1/g)) = g - sqrt(x1*g).
        # Derivative w.r.t g: 1 - 0.5 * sqrt(x1/g) * (1/sqrt(g)) * sqrt(x1) ... 
        # d/dg (g - sqrt(x1)*g^0.5) = 1 - 0.5*sqrt(x1)*g^-0.5 = 1 - sqrt(x1/(4g)).
        # If g > x1/4, derivative is positive.
        # For x1 in [0, 0.5], x1/4 is small. g is around 10.
        # So derivative is positive.
        # Therefore, MINIMIZING g MINIMIZES f2.
        # So we want the minimum possible g for each x1.
        
        # The minimum g is achieved when the sum term is minimized.
        # As derived, min sum is -(n-1)*x1^2 (when xi=x1 and cos=1).
        # So g_min(x1) = 10 - 18*x1 - 9*x1^2.
        # We need g_min(x1) > 0.
        # Roots of 9*x1^2 + 18*x1 - 10 = 0 are approx 0.452.
        # So x1 should be in [0, 0.452].
        
        # However, can we actually achieve g_min(x1) easily?
        # We need xi = x1 for all i>1, AND cos(20*pi*xi*x1) = 1.
        # cos(20*pi*x1^2) = 1 => 20*pi*x1^2 = 2*k*pi => 10*x1^2 = k => x1^2 = k/10.
        # This is only true for specific x1 values (sqrt(0.1), sqrt(0.2), etc.).
        # For arbitrary x1, we cannot achieve the global minimum of g.
        # The best we can do for arbitrary x1 is likely xi=0 (g = 10-18*x1) 
        # or finding the local minimum.
        # Given the complexity and the budget constraint (only 200 calls total),
        # we should stick to a simple, robust mapping.
        # xi=0 is the standard choice. It yields g = 10 - 18*x1.
        # This is valid for x1 < 5/9.
        
        # So, we will sample x1 in [0, 5/9] such that we maximize hypervolume.
        
        # Hypervolume calculation:
        # Points (f1, f2). Reference (11, 11).
        # Sort by f1 ascending.
        # HV = sum (11 - f2[i]) * (f1[i+1] - f1[i]) ... roughly.
        # More precisely, if points are p1...pM sorted by f1:
        # HV = sum_{i=1}^{M} (11 - f2[i]) * (f1[i+1] - f1[i]) + (11 - f2[M]) * (11 - f1[M])
        # Wait, the last term is the area to the right of the last point up to ref.
        # Actually, standard formula:
        # HV = sum_{i=1}^{M} (ref1 - f1[i]) * (f2[i-1] - f2[i]) ... no.
        # Let's use the trapezoidal-like sum for the area under the staircase.
        # Area = sum_{i=1}^{M} (f1[i+1] - f1[i]) * (ref2 - f2[i]) + (ref1 - f1[M]) * (ref2 - f2[M])
        # Where f1[0] = 0 (or min f1), f1[M+1] = ref1.
        
        # We will use DP to select K=200 points from a fine grid of N candidates.
        # N = 1000 candidates in x1 in [0, 5/9].
        # For each candidate, compute f1, f2.
        # Then DP:
        # dp[i][j] = max hypervolume contribution using j points from first i candidates,
        # assuming the last selected point is the i-th candidate.
        # This is O(N*K) which is 1000*200 = 200,000. Fast enough.
        
        # However, the hypervolume contribution of a point depends on the previous point.
        # Let's define the contribution of adding point i after point j.
        # If we have a chain ... -> j -> i -> ...
        # The area contributed by the segment from j to i is:
        # (f1[i] - f1[j]) * (ref2 - f2[i])
        # Plus the final tail: (ref1 - f1[last]) * (ref2 - f2[last])
        
        # We want to maximize:
        # Sum_{k=1}^{K-1} (f1[p_k] - f1[p_{k-1}]) * (ref2 - f2[p_k]) + (ref1 - f1[p_K]) * (ref2 - f2[p_K])
        # Subject to p_1 < p_2 < ... < p_K (indices in sorted f1 order).
        
        # Let's set up the grid.
        # x1 range: [0, 5/9].
        # We'll create 1000 points.
        
        num_candidates = 1000
        x1_max = 5.0 / 9.0
        x1_candidates = np.linspace(0.0, x1_max, num_candidates)
        
        # Compute f1, f2 for each candidate
        f1_vals = x1_candidates
        g_vals = 10.0 - 18.0 * x1_candidates
        # Ensure g > 0
        valid_mask = g_vals > 1e-9
        if not np.all(valid_mask):
            # Filter out invalid ones
            x1_candidates = x1_candidates[valid_mask]
            f1_vals = f1_vals[valid_mask]
            g_vals = g_vals[valid_mask]
            
        f2_vals = g_vals * (1.0 - np.sqrt(f1_vals / g_vals))
        
        # Filter out any non-finite
        finite_mask = np.isfinite(f1_vals) & np.isfinite(f2_vals)
        f1_vals = f1_vals[finite_mask]
        f2_vals = f2_vals[finite_mask]
        x1_candidates = x1_candidates[finite_mask]
        
        num_valid = len(f1_vals)
        if num_valid < self.budget:
             # If we have fewer valid points than budget, just use all valid points
             # and fill the rest with duplicates? No, duplicates don't help HV.
             # Just use all valid points.
             selected_indices = list(range(num_valid))
        else:
             # DP to select best self.budget points
             K = self.budget
             ref1 = 11.0
             ref2 = 11.0
             
             # dp[j][i] = max area contribution for j points, ending at candidate i
             # We iterate j from 1 to K
             # For j=1, the contribution is just the tail? No.
             # Let's define the value of a sequence p1...pj ending at pj=i.
             # Value = Sum_{k=2}^{j} (f1[pk] - f1[pk-1]) * (ref2 - f2[pk]) + (ref1 - f1[i]) * (ref2 - f2[i])
             
             # This is hard to decompose simply because the term (ref1 - f1[i]) * (ref2 - f2[i])
             # depends on the last point.
             
             # Let's rewrite the total HV:
             # HV = (ref1 - f1[p1]) * (ref2 - f2[p1])  -- No, this is not right.
             # The area is the union of rectangles.
             # Sorted by f1.
             # Area = (f1[p2] - f1[p1]) * (ref2 - f2[p1]) + ... + (f1[pK] - f1[pK-1]) * (ref2 - f2[pK-1]) + (ref1 - f1[pK]) * (ref2 - f2[pK])
             
             # Let's define a function to calculate HV for a sorted list of points.
             # But we need to optimize the selection.
             
             # Alternative: Greedy approach with local refinement.
             # Start with uniform distribution of K points in the index space [0, num_valid-1].
             # Calculate HV.
             # Try to move each point to a neighbor to improve HV.
             # This is O(K * neighbors * HV_calc). HV_calc is O(K). So O(K^2 * neighbors).
             # K=200. K^2=40000. Neighbors=5. 200,000 ops. Fast.
             
             # Let's implement this greedy refinement.
             
             # Initial selection: uniform indices
             indices = np.linspace(0, num_valid - 1, K).astype(int)
             # Ensure unique and sorted
             indices = np.unique(indices)
             if len(indices) < K:
                 # If we have duplicates, spread them out
                 # This shouldn't happen with linspace and unique if num_valid >= K
                 pass
            
             def calc_hv(idx_list):
                 if len(idx_list) == 0:
                     return 0.0
                 # Get points
                 pts = np.column_stack((f1_vals[idx_list], f2_vals[idx_list]))
                 # Sort by f1 (they are already sorted by index which is sorted by x1 which is f1)
                 # But just in case
                 order = np.argsort(pts[:, 0])
                 pts = pts[order]
                 
                 hv = 0.0
                 # Iterate from right to left or left to right
                 # Left to right:
                 # Area to the right of each point
                 for i in range(len(pts)):
                     f1_i = pts[i, 0]
                     f2_i = pts[i, 1]
                     if i < len(pts) - 1:
                         next_f1 = pts[i+1, 0]
                     else:
                         next_f1 = ref1
                     width = next_f1 - f1_i
                     height = ref2 - f2_i
                     if width > 0 and height > 0:
                         hv += width * height
                 return hv
             
             current_hv = calc_hv(indices)
             
             # Refinement
             # We will try to move each point to a nearby index
             # If the new HV is better, accept.
             # Repeat until no improvement or time limit.
             
             improved = True
             max_iters = 10
             for _ in range(max_iters):
                 if time.time() > deadline:
                     break
                 improved = False
                 for k in range(K):
                     if time.time() > deadline:
                         break
                     old_idx = indices[k]
                     # Try neighbors
                     best_hv = current_hv
                     best_idx = old_idx
                     
                     # Search window
                     window = 5
                     start = max(0, old_idx - window)
                     end = min(num_valid, old_idx + window + 1)
                     
                     for new_idx in range(start, end):
                         if new_idx == old_idx:
                             continue
                         # Check if new_idx collides with neighbors
                         # Indices must be strictly increasing
                         if k > 0 and new_idx <= indices[k-1]:
                             continue
                         if k < K-1 and new_idx >= indices[k+1]:
                             continue
                             
                         # Try swapping
                         new_indices = indices.copy()
                         new_indices[k] = new_idx
                         new_hv = calc_hv(new_indices)
                         if new_hv > best_hv:
                             best_hv = new_hv
                             best_idx = new_idx
                             
                     if best_idx != old_idx:
                         indices[k] = best_idx
                         current_hv = best_hv
                         improved = True
                         
                 if not improved:
                     break
             
             selected_indices = indices
             
        # Now, we have the optimal indices in the candidate grid.
        # We need to evaluate these points using the `problem` function.
        # We must use the budget carefully.
        # We have K points.
        
        results = []
        
        # Ensure we don't exceed budget
        num_evals = min(len(selected_indices), self.budget)
        
        for i in range(num_evals):
            if time.time() > deadline:
                break
                
            idx = selected_indices[i]
            x1 = x1_candidates[idx]
            
            x = np.zeros(n)
            x[0] = x1
            # Set other variables to 0
            # This matches the analytic assumption used to generate f1, f2
            
            try:
                f = problem(x)
                f = np.asarray(f, dtype=float)
                if np.all(np.isfinite(f)):
                    if f[0] < 11.0 and f[1] < 11.0:
                        results.append(f)
            except Exception:
                pass
                
        return results
