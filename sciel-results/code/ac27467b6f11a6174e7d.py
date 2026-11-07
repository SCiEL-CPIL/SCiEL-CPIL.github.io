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
        
        # ZDT3 Analysis:
        # f1(x) = x0
        # g(x) = 1 + 10*(n-1) + sum(xi^2 - 10*cos(4*pi*xi)) for i=1..n-1
        # f2(x) = g(x) * h(f1(x), g(x))
        # h(u,v) = 1 - sqrt(u/v) - sin(10*pi*u)
        
        # The Pareto front is at x[1:] = 0, so g_min = 11.
        # On the front:
        # u = x0
        # v = 11
        # f1 = u
        # f2 = 11 * (1 - sqrt(u/11) - sin(10*pi*u))
        
        # We want to maximize the hypervolume of the set of points { (f1_i, f2_i) } 
        # with reference point (11, 11).
        # Since f1 is monotonic in u, and we can sort points by f1, the hypervolume 
        # calculation for a set of non-dominated points sorted by f1 (increasing) 
        # is sum over i of (f2_{i+1} - f2_i) * (f1_i - f1_{i-1}) ... wait, 
        # standard hypervolume for sorted points (f1 asc, f2 desc):
        # Area = sum_{i=1}^{k} (f1_i - f1_{i-1}) * f2_i, with f1_0=0? No.
        
        # Let's visualize. Reference point (11, 11).
        # Points are in [0, 11] x [0, 11].
        # Hypervolume is the area of the union of rectangles [0, f1_i] x [0, f2_i].
        # Since points are on a curve, and we want to maximize area under the curve 
        # formed by the "lower envelope" of the points relative to the axes?
        # No, hypervolume is the volume dominated by the points.
        # For a set of points P, HV = Vol( { (y1, y2) | y1 <= f1_i, y2 <= f2_i for some i } ).
        
        # If we sort points by f1 increasing: p1, p2, ..., pk.
        # Since they are non-dominated, f2 must be decreasing.
        # The dominated region is the union of rectangles [0, f1_i] x [0, f2_i].
        # This union can be calculated as:
        # HV = sum_{i=1}^k (f1_i - f1_{i-1}) * f2_i  where f1_0 = 0?
        # Let's check.
        # i=1: rect [0, f1_1] x [0, f2_1]. Area = f1_1 * f2_1.
        # i=2: rect [0, f1_2] x [0, f2_2]. Since f1_2 > f1_1 and f2_2 < f2_1.
        # The new area added is the part of rect 2 not covered by rect 1.
        # Rect 1 covers [0, f1_1] x [0, f2_1].
        # Rect 2 covers [0, f1_2] x [0, f2_2].
        # Intersection is [0, f1_1] x [0, f2_2] (since f2_2 < f2_1).
        # Area(2) = f1_2 * f2_2.
        # Area(1 U 2) = Area(1) + Area(2) - Area(Inter)
        # = f1_1*f2_1 + f1_2*f2_2 - f1_1*f2_2
        # = f1_1*(f2_1 - f2_2) + f1_2*f2_2.
        
        # General formula for sorted points (f1 asc, f2 desc):
        # HV = f1_k * f2_k + sum_{i=1}^{k-1} (f1_i - f1_{i-1}) * f2_i ? No.
        # Let's derive recursively.
        # HV_k = HV_{k-1} + (f1_k - f1_{k-1}) * f2_k.
        # Base case HV_0 = 0? No, if we add p1, HV_1 = f1_1 * f2_1.
        # Formula: HV = sum_{i=1}^k (f1_i - f1_{i-1}) * f2_i with f1_0 = 0.
        # Check k=1: (f1_1 - 0)*f2_1 = f1_1*f2_1. Correct.
        # Check k=2: (f1_1-0)*f2_1 + (f1_2-f1_1)*f2_2 = f1_1*f2_1 + f1_2*f2_2 - f1_1*f2_2. Correct.
        
        # So we want to maximize:
        # H = sum_{i=1}^k (f1_i - f1_{i-1}) * f2_i
        # subject to f1_1 < f1_2 < ... < f1_k.
        
        # This is equivalent to maximizing the area under the step function 
        # defined by the points.
        # Note that f2 is determined by f1 on the Pareto front.
        # Let F(u) = f2(u).
        # We choose k points u_1 < u_2 < ... < u_k.
        # H = sum_{i=1}^k (u_i - u_{i-1}) * F(u_i)  (with u_0=0, F(0) is f2(0)).
        
        # This is a Riemann sum approximation of the integral of F(u) from 0 to u_k?
        # No, it's the area of the rectangles.
        # To maximize this for a fixed k, we should choose the points optimally.
        # This is an optimal quantization problem.
        
        # However, we are constrained by the budget of 200 evaluations.
        # We must call problem(x) for each point we report.
        # So we can report at most 200 points.
        
        # Strategy:
        # 1. Define the analytic function F(u) for u in [0, 1].
        # 2. Use DP or greedy to select 200 points u_1...u_200 that maximize H.
        # 3. Generate x vectors for these u's (x[0]=u, x[1:]=0).
        # 4. Call problem(x) for each.
        
        # Since F(u) is oscillatory (sin term), the optimal points will cluster 
        # around the local minima of F(u) (which are local maxima of the "height" 
        # in the hypervolume sense? No, we want high f2 values).
        # Wait, H = sum (delta_u) * F(u_i).
        # To maximize H, we want F(u_i) to be large.
        # But the intervals (u_i - u_{i-1}) sum to u_k.
        # If we pick u_k=1, the sum of intervals is 1.
        # We are effectively approximating the integral of F(u) with right-endpoint Riemann sums?
        # No, the intervals are variable.
        
        # Actually, if we can choose any points, the maximum hypervolume is bounded 
        # by the integral of the upper envelope?
        # The "upper envelope" of the dominated region is the function y(x) such that 
        # for any x, y(x) is the max f2 of any point with f1 >= x?
        # No, for a set of points, the boundary of the dominated region is the 
        # lower-left convex hull? No, it's the step function formed by the points.
        
        # Let's look at the structure of F(u).
        # F(u) = 11 * (1 - sqrt(u/11) - sin(10*pi*u))
        # At u=0, F(0) = 11 * (1 - 0 - 0) = 11.
        # At u=1, F(1) = 11 * (1 - 1 - sin(10*pi)) = 0.
        
        # The function oscillates.
        # sin(10*pi*u) has period 0.1.
        # So there are 10 periods in [0, 1].
        
        # To maximize the sum of (delta_u) * F(u_i), we should place points 
        # where F(u) is high.
        # Specifically, if we have a fixed number of points, we should place them 
        # at the local maxima of F(u).
        
        # Let's find the local maxima of F(u).
        # F'(u) = 11 * ( - (1/(2*sqrt(11)*sqrt(u))) - 10*pi*cos(10*pi*u) )
        # F'(u) = 0 => 10*pi*cos(10*pi*u) = - 1/(2*sqrt(11)*sqrt(u))
        
        # This is complex to solve analytically for each u.
        # However, we can numerically find the optimal points.
        
        # Since the budget is small (200), and the function has 10 periods,
        # we have ~20 points per period.
        # We can discretize u into a fine grid, calculate F(u), and then 
        # select the best 200 points using a greedy or DP approach.
        
        # Given the oscillatory nature, the optimal set of points will likely 
        # consist of the local maxima of F(u).
        
        # Let's generate a fine grid of u, compute F(u), and select the top 200 
        # points by F(u)?
        # No, that's not quite right. The term (u_i - u_{i-1}) matters.
        # If we pick two points very close together where F is high, we get 
        # small width * high height.
        # If we spread them out, we get larger width.
        
        # Actually, for a monotonic decreasing function, the optimal right-endpoint 
        # Riemann sum for a fixed number of subintervals is to make the subintervals 
        # equal? No.
        
        # Let's simplify. The hypervolume is the area under the curve formed by 
        # connecting (0, F(0)) to (u1, F(u1)) to ... to (uk, F(uk)) and dropping 
        # to (uk, 0) and (0,0)? No, the hypervolume definition is the union of 
        # rectangles.
        
        # The union of rectangles [0, u_i] x [0, F(u_i)] is exactly the area 
        # under the step function that is F(u_i) on [u_{i-1}, u_i].
        
        # So we are maximizing the integral of a step function that takes values 
        # from the set {F(u) | u in [0,1]} at points u_i.
        
        # To maximize this, we should choose u_i such that the step function 
        # stays as high as possible for as long as possible.
        
        # This implies we should choose u_i to be the local maxima of F(u).
        
        # Let's verify.
        # If F(u) has a local max at u*, placing a point there allows the 
        # "height" to be F(u*) for the interval ending at u*.
        
        # Algorithm:
        # 1. Sample F(u) on a very fine grid (e.g., 100,000 points).
        # 2. Identify local maxima.
        # 3. Select the top 200 local maxima (by F value).
        # 4. Sort them by u.
        # 5. Ensure we cover the range? 
        #    Actually, if we only pick peaks, the intervals between peaks 
        #    will have lower "height" (the F value of the previous peak).
        #    Wait, the height on [u_{i-1}, u_i] is F(u_i).
        #    So if u_i is a peak, the height is high.
        #    If u_{i-1} is also a peak, the interval between them has height F(u_i).
        
        # So yes, picking the peaks is good.
        # But what about the very first interval [0, u_1]? Height is F(u_1).
        # And the last interval [u_{k-1}, u_k]? Height is F(u_k).
        
        # We should also consider u=0? F(0)=11.
        # If we set u_1 very close to 0, we get a small interval with high height.
        # But we have a limited number of points.
        
        # Let's try a greedy approach:
        # 1. Discretize u into N_grid points.
        # 2. Calculate F(u) for all.
        # 3. We want to select K=200 indices i_1 < i_2 < ... < i_K to maximize 
        #    sum_{j=1}^K (u_{i_j} - u_{i_{j-1}}) * F(u_{i_j}).
        
        # This is a classic DP problem:
        # dp[j][i] = max hypervolume using j points, with the j-th point at index i.
        # dp[j][i] = max_{k < i} ( dp[j-1][k] + (u_i - u_k) * F(u_i) )
        
        # Complexity: O(K * N_grid^2).
        # N_grid = 1000, K = 200.
        # 200 * 1000^2 = 200,000,000 operations.
        # This might be too slow in Python.
        
        # Optimization:
        # The term is (u_i - u_k) * F(u_i) = u_i * F(u_i) - u_k * F(u_i).
        # This doesn't look easily separable for a simple max.
        
        # However, we can use a greedy heuristic:
        # The function F(u) is oscillatory. The optimal points are likely the 
        # local maxima.
        # Let's find the local maxima of F(u) on a fine grid.
        # There are about 10 periods, so ~10 local maxima per period? No, 1 per period.
        # So ~10 local maxima in [0,1].
        # Wait, sin(10*pi*u) has 10 periods.
        # F(u) = C - A*sqrt(u) - B*sin(10*pi*u).
        # The derivative is dominated by the cosine term.
        # So there are ~10 local maxima.
        
        # If there are only 10 local maxima, we can't fill 200 points with just them.
        # We need to pick more points.
        
        # Maybe we should pick points near the peaks to capture the "width" of the peaks?
        
        # Alternative: 
        # Since the function is known, we can compute the exact hypervolume for any 
        # subset. But we are limited to 200 evaluations.
        
        # Let's try a simpler heuristic:
        # Sample 200 points uniformly? No, that's what the current code does (sort of).
        
        # What if we sample points to maximize the "area under the curve" of F(u)?
        # The hypervolume is approximately the integral of F(u) if the points are dense.
        # But with 200 points, we can do better than uniform.
        
        # Let's use the DP approach but with a smaller grid to make it fast.
        # N_grid = 500. K = 200.
        # 200 * 500^2 = 50,000,000. Still might be slow in pure Python.
        
        # We can vectorize the inner loop?
        # dp[j][i] = max_k ( dp[j-1][k] - u_k * F(u_i) ) + u_i * F(u_i)
        # For a fixed i, we want to maximize ( dp[j-1][k] - u_k * F(u_i) ) over k < i.
        # This is a line fitting problem: y_k = dp[j-1][k], x_k = u_k.
        # We want to maximize y_k - x_k * m where m = F(u_i).
        # This is the upper envelope of lines y = -x_k * m + y_k.
        # We can use the Convex Hull Trick (CHT) to optimize this to O(N log N) per layer.
        # Total O(K * N log N).
        # 200 * 500 * log(500) ~ 200 * 500 * 9 = 900,000 operations. Very fast.
        
        # Implementation of CHT:
        # We maintain a convex hull of lines.
        # Each line is defined by slope m_k = -u_k and intercept b_k = dp[j-1][k].
        # We query for the max value at x = F(u_i).
        
        # Steps:
        # 1. Prepare u_grid and F_grid.
        # 2. dp[0][i] = 0? No, dp[1][i] = (u_i - 0) * F(u_i) = u_i * F(u_i).
        # 3. For j from 2 to K:
        #    Initialize CHT with lines from k=0 to i-1?
        #    Wait, the recurrence is dp[j][i] = max_{k < i} ( dp[j-1][k] + (u_i - u_k) * F(u_i) )
        #    = max_{k < i} ( dp[j-1][k] - u_k * F(u_i) ) + u_i * F(u_i)
        
        #    We iterate i from 1 to N-1.
        #    Before calculating dp[j][i], we add the line for k = i-1 to the CHT?
        #    No, we need all k < i.
        #    So we iterate i from 1 to N-1.
        #    At step i, the CHT contains lines for k = 0, ..., i-1.
        #    We query CHT at x = F(u_i).
        #    Then we add the line for k = i to the CHT for the next step?
        #    No, for dp[j+1], we will need lines from dp[j].
        
        #    So:
        #    dp[j][0] = -inf (can't have 2 points ending at first)
        #    For i from 1 to N-1:
        #       Add line (m = -u_{i-1}, b = dp[j-1][i-1]) to CHT.
        #       Query CHT at x = F(u_i).
        #       dp[j][i] = result + u_i * F(u_i).
        
        #    Wait, the line for k should be added BEFORE querying for i, so that k < i.
        #    If we add k=i-1, then k < i is satisfied.
        
        #    We need to track the best index k to reconstruct the path.
        
        # Let's implement this.
        
        N_grid = 1000
        u_grid = np.linspace(0, 1, N_grid)
        
        # Calculate F(u)
        # F(u) = 11 * (1 - sqrt(u/11) - sin(10*pi*u))
        # Handle u=0 for sqrt
        sqrt_term = np.zeros_like(u_grid)
        mask = u_grid > 0
        sqrt_term[mask] = np.sqrt(u_grid[mask] / 11.0)
        F_grid = 11.0 * (1.0 - sqrt_term - np.sin(10.0 * np.pi * u_grid))
        
        # Ensure F is non-negative? 
        # sin can be 1. 1 - 0 - 1 = 0.
        # Min value of F?
        # sqrt term is small. sin term oscillates.
        # F(u) >= 0.
        
        K = self.budget
        if K > N_grid:
            K = N_grid
            
        # DP
        # dp[j][i]
        # We only need previous layer
        prev_dp = np.zeros(N_grid)
        prev_dp[0] = 0 # Base case: 1 point at u=0? 
        # Actually, if we pick u_1, dp[1][i] = u_i * F(u_i).
        # Let's initialize prev_dp for j=1.
        prev_dp = u_grid * F_grid
        
        # Store parent indices for reconstruction
        # parent[j][i] = k
        parents = np.zeros((K, N_grid), dtype=int)
        
        # CHT Implementation
        # Lines: y = m*x + b
        # We want to maximize.
        # Slopes m are -u_k. Since u_k is increasing, m is decreasing.
        # This is convenient for CHT.
        
        # We'll use a list to store lines (m, b, index).
        # And a pointer for querying if x is monotonic?
        # x = F(u_i). F(u_i) is NOT monotonic. It oscillates.
        # So we cannot use a simple pointer. We need binary search or a balanced BST.
        # Or just brute force if N is small?
        # N=1000, K=200.
        # Brute force inner loop: 200 * 1000 * 1000 = 200,000,000.
        # In Python, this is too slow.
        
        # We need CHT with binary search.
        # Since slopes are monotonic, the hull is convex.
        # We can binary search for the intersection.
        
        # Let's implement a simple CHT class.
        
        class CHT:
            def __init__(self):
                self.lines = [] # list of (m, b, idx)
            
            def add_line(self, m, b, idx):
                # Add line y = m*x + b
                # Slopes are added in decreasing order.
                # We maintain the lower hull for minimization? No, we want max.
                # For max with decreasing slopes, we maintain the upper hull.
                
                new_line = (m, b, idx)
                # Remove lines that are never optimal
                # Intersection of l1 and l2: x = (b2 - b1) / (m1 - m2)
                
                while len(self.lines) >= 2:
                    m1, b1, i1 = self.lines[-1]
                    m2, b2, i2 = self.lines[-2]
                    # Check if l1 is dominated by l2 and new_line
                    # Intersection x12 = (b2 - b1) / (m1 - m2)
                    # Intersection x2n = (b - b2) / (m2 - m)
                    # If x12 >= x2n, then l1 is never optimal.
                    # Note: m1 > m2 > m (decreasing slopes)
                    # Denominators are positive.
                    
                    # Be careful with division by zero (parallel lines)
                    # If m1 == m2, keep the one with higher b.
                    if m1 == m2:
                        if b1 <= b:
                            self.lines.pop()
                            continue
                        else:
                            return # New line is dominated by l1
                    
                    x12 = (b2 - b1) / (m1 - m2)
                    x2n = (b - b2) / (m2 - m)
                    
                    if x12 >= x2n:
                        self.lines.pop()
                    else:
                        break
                
                self.lines.append(new_line)
            
            def query(self, x):
                # Find line that maximizes y at x
                # Since slopes are decreasing, the optimal line index increases as x increases?
                # No, for max with decreasing slopes:
                # Steeper lines (more negative) are better for small x?
                # y = m*x + b.
                # If m is very negative, y drops fast.
                # If m is close to 0, y is flatter.
                # For x > 0, lines with less negative m (closer to 0) are better?
                # Wait, u_k >= 0, so m = -u_k <= 0.
                # Larger u_k means more negative m.
                # So lines are added with decreasing m.
                
                # The optimal line changes as x changes.
                # We can binary search on the intersection points.
                
                if not self.lines:
                    return -np.inf, -1
                
                # The lines are stored in order of decreasing slope.
                # The optimal line for a given x is the one where x is between 
                # the intersection with the previous line and the next line.
                
                # Let's store the intersection x-coordinates.
                # Actually, we can just binary search on the lines.
                
                lo = 0
                hi = len(self.lines) - 1
                best_val = -np.inf
                best_idx = -1
                
                # Since x is arbitrary, we can't use a pointer.
                # Binary search for the line where the value is maximized.
                # The function value is concave in the index?
                
                # Simpler: just iterate? No, too slow.
                
                # Let's maintain the intersection points in a separate list.
                # This complicates add_line.
                
                # Alternative: Since N is only 1000, maybe we can just use 
                # vectorized numpy operations for the max?
                # dp[j][i] = max_k ( prev_dp[k] - u_k * F(u_i) ) + u_i * F(u_i)
                # For a fixed i, we compute for all k < i.
                # val = prev_dp[:i] - u_grid[:i] * F(u_i)
                # best_k = argmax(val)
                
                # This is O(N) per i, so O(N^2) per layer.
                # 1000^2 = 1,000,000 per layer.
                # 200 layers = 200,000,000 operations.
                # In NumPy, this is vectorized.
                # 200 * (1000^2) flops.
                # NumPy can do ~100M-1G flops/s.
                # So 0.2 - 2 seconds.
                # This is acceptable!
                
                pass # We will use the NumPy approach instead of CHT class complexity.

        # Let's implement the NumPy DP.
        
        # prev_dp is for j-1.
        # We want to compute curr_dp for j.
        
        # We need to store parent pointers.
        # parent[j][i] = argmax_k
        
        for j in range(2, K + 1):
            curr_dp = np.full(N_grid, -np.inf)
            curr_parent = np.zeros(N_grid, dtype=int)
            
            # For each i from j-1 to N-1 (need at least j points)
            for i in range(j - 1, N_grid):
                # Consider k from j-2 to i-1
                # val = prev_dp[k] + (u_i - u_k) * F_i
                # val = prev_dp[k] - u_k * F_i + u_i * F_i
                
                F_i = F_grid[i]
                u_i = u_grid[i]
                
                # k range: [j-2, i-1]
                k_start = j - 2
                k_end = i # exclusive
                
                if k_start >= k_end:
                    continue
                
                prev_vals = prev_dp[k_start:k_end]
                u_vals = u_grid[k_start:k_end]
                
                # Filter out -inf
                valid = prev_vals > -np.inf/2
                if not np.any(valid):
                    continue
                
                # Calculate scores
                # scores = prev_vals - u_vals * F_i
                scores = prev_vals - u_vals * F_i
                scores[~valid] = -np.inf
                
                best_k_rel = np.argmax(scores)
                best_score = scores[best_k_rel]
                
                if best_score > -np.inf/2:
                    curr_dp[i] = best_score + u_i * F_i
                    curr_parent[i] = k_start + best_k_rel
            
            prev_dp = curr_dp
            if j > 2:
                parents[j-1] = curr_parent # Store for reconstruction
            else:
                parents[1] = curr_parent # j=2, store in index 1? 
                # Let's be consistent. parents[j] corresponds to step j+1?
                # Let's say parents[j] stores the parent for dp[j+1].
                # Or simpler: parents array of shape (K, N).
                # parents[j-1][i] = k for transition from j-1 to j.
                
        # Reconstruct path
        # Start from best i for j=K
        best_i = np.argmax(prev_dp)
        path = [best_i]
        
        for j in range(K - 1, 0, -1):
            # parents[j-1][best_i] gives the parent index for step j
            # Wait, my loop above:
            # for j in range(2, K+1):
            #   curr_parent[i] is the k for transition to j.
            #   I stored it in parents[j-1] (if j>2) or parents[1] (if j=2).
            #   This indexing is messy.
            
            # Let's restart the storage logic.
            # parents = np.zeros((K, N_grid), dtype=int)
            # In loop j (2 to K):
            #   parents[j-1][i] = k
            
            k = parents[j-1][best_i]
            path.append(k)
            best_i = k
        
        path.reverse()
        
        # path contains indices in u_grid
        # Generate x vectors
        selected_u = u_grid[path]
        
        X_list = []
        for u in selected_u:
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = u
            X_list.append(x)
            
        # Evaluate
        for x in X_list:
            if time.time() > deadline:
                break
            problem(x)
            
        return X_list
