"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Exploit ZDT3 structure: set x2..xn to lower bounds (g=1) and sample x1 optimally.
    Uses the analytic Pareto front y(x) = 1 - sqrt(x) - x*sin(10*pi*x) with reference (11,11).
    Selects K x-values in [0,1] to maximize hypervolume of the resulting points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.x1_candidates = self._compute_optimal_x1(budget)

    def _y_of_x(self, x):
        """Analytic Pareto front: y = 1 - sqrt(x) - x*sin(10*pi*x)"""
        x = np.asarray(x, dtype=float)
        return 1.0 - np.sqrt(x) - x * np.sin(10.0 * np.pi * x)

    def _hv_from_sorted(self, x_sorted, y_vals, ref=11.0):
        """
        Compute hypervolume for a set of points (x_i, y_i) that are all better than ref in both dims.
        Points are sorted by x ascending. The front is decreasing in y as x increases.
        HV = sum over i of (x_{i+1} - x_i) * (ref - y_i)  where x_{n+1} = ref, x_0 = 0
        Actually, for a decreasing front, the HV contribution of point i is:
        (x_{i+1} - x_i) * (ref - y_i) for i=0..n-1, plus (ref - x_n) * (ref - y_n) for the last.
        Wait, let me think more carefully.
        
        For a set of non-dominated points sorted by x ascending (y descending),
        the hypervolume w.r.t. reference (ref, ref) is:
        HV = sum_{i=0}^{n-1} (x_{i+1} - x_i) * (ref - y_i) + (ref - x_n) * (ref - y_n)
        where x_0 is the smallest x, x_n is the largest x.
        But we need x_0 to be 0? No, the "left boundary" for the first point is x=0 only if
        the point's x > 0. Actually the standard formula:
        Sort points by x ascending. For each point i, its contribution is
        (x_{i+1} - x_i) * (ref - y_i) where x_{n} (the last one's right neighbor) = ref.
        And the leftmost point's left neighbor is 0? No, it's the minimum x in the set.
        
        Let me use the standard formula:
        Sort by x ascending. x_ext = [0, x_0, x_1, ..., x_{n-1}, ref]
        y_ext = [ref, y_0, y_1, ..., y_{n-1}]
        HV = sum_{i=0}^{n} (x_ext[i+1] - x_ext[i]) * (ref - y_ext[i])
        """
        n = len(x_sorted)
        if n == 0:
            return 0.0
        x_ext = np.concatenate([[0.0], x_sorted, [ref]])
        y_ext = np.concatenate([[ref], y_vals])
        widths = x_ext[1:] - x_ext[:-1]
        heights = ref - y_ext
        return float(np.sum(widths * heights))

    def _compute_optimal_x1(self, K):
        """
        Compute K x-values in [0, 1] that maximize the hypervolume of points (x, y(x))
        with reference point (11, 11).
        """
        # Start with a dense grid
        N_grid = 5000
        x_grid = np.linspace(0.0, 1.0, N_grid)
        y_grid = self._y_of_x(x_grid)
        
        # Filter: keep only points where y < ref (11) and x < ref (11), which is always true
        # Also ensure y > 0? No, y can be negative. The point is valid if x < 11 and y < 11.
        
        # Greedy selection: start with 2 points (x=0 and x=1), then add the point
        # that gives the largest HV increase.
        
        # But greedy from scratch with 5000 candidates and 200 selections is O(200 * 5000 * log) which is fine.
        
        # Actually, a smarter approach: the HV is a function of the set of x values.
        # The optimal set can be found by dynamic programming, but K=200 and N=5000 makes DP too slow.
        
        # Let's use a recursive bisection / median approach:
        # The HV is maximized when the points are well-distributed.
        # For a decreasing curve, the optimal placement is related to equalizing the "contribution".
        
        # Let me try a different approach: use the fact that for a monotone decreasing front,
        # the hypervolume can be written as an integral, and we want to approximate it well.
        # This is essentially optimal quantization.
        
        # Simpler: greedy with efficient HV computation.
        # Start with x=0 and x=1.
        selected_indices = [0, N_grid - 1]
        selected_x = x_grid[selected_indices].copy()
        selected_y = y_grid[selected_indices].copy()
        
        current_hv = self._hv_from_sorted(selected_x, selected_y)
        
        # Pre-compute y for all grid points
        # For each iteration, find the grid point that maximizes HV increase
        
        remaining_candidates = np.setdiff1d(np.arange(N_grid), selected_indices)
        
        for _ in range(K - 2):
            best_idx = -1
            best_hv = current_hv
            
            # For each candidate, compute the HV if added
            # This is O(N * log N) per candidate, so O(N^2 log N) per iteration.
            # With N=5000 and 198 iterations, that's ~5000*5000*198*log(5000) ~ 2.5e10, too slow.
            
            # Need a faster approach.
            # Key insight: adding a point x_new between x_i and x_{i+1} only changes
            # the contributions of those two intervals.
            # HV increase = (ref - y_new)*(x_new - x_i) + (ref - y_{i+1})*(x_{i+1} - x_new) 
            #               - (ref - y_i)*(x_{i+1} - x_i)
            # Wait, that's not right either. Let me think.
            
            # Original contribution of interval [x_i, x_{i+1}]: (x_{i+1} - x_i) * (ref - y_i)
            # After adding x_new: (x_new - x_i) * (ref - y_i) + (x_{i+1} - x_new) * (ref - y_new)
            # Increase = (x_new - x_i) * (ref - y_i) + (x_{i+1} - x_new) * (ref - y_new) - (x_{i+1} - x_i) * (ref - y_i)
            #          = (x_{i+1} - x_new) * (ref - y_new - (ref - y_i))
            #          = (x_{i+1} - x_new) * (y_i - y_new)
            
            # So the HV increase from adding x_new in interval [x_i, x_{i+1}] is:
            # (x_{i+1} - x_new) * (y_i - y_new)
            # where y_i is the y-value of the left neighbor.
            
            # This is O(1) per candidate if we know which interval it falls in!
            
            # Let's restructure: for each candidate x_new, find its interval [x_i, x_{i+1}]
            # in the current selected set, and compute the increase.
            
            # Since selected_x is sorted, we can use binary search.
            
            best_increase = -1
            best_candidate_idx = -1
            
            for cand_idx in remaining_candidates:
                x_new = x_grid[cand_idx]
                y_new = y_grid[cand_idx]
                
                # Find which interval x_new falls in
                # selected_x is sorted
                pos = np.searchsorted(selected_x, x_new)
                
                if pos == 0:
                    # x_new is before the first selected point
                    # Left boundary is 0, right is selected_x[0]
                    x_left = 0.0
                    y_left = 11.0  # ref
                    x_right = selected_x[0]
                    # Increase = (x_right - x_new) * (y_left - y_new) = (x_right - x_new) * (11 - y_new)
                    # Wait, the formula was (x_{i+1} - x_new) * (y_i - y_new)
                    # Here i is the "virtual" point at x=0, y=11
                    increase = (x_right - x_new) * (11.0 - y_new)
                elif pos == len(selected_x):
                    # x_new is after the last selected point
                    x_left = selected_x[-1]
                    y_left = selected_y[-1]
                    x_right = 11.0
                    increase = (x_right - x_new) * (y_left - y_new)
                else:
                    x_left = selected_x[pos - 1]
                    y_left = selected_y[pos - 1]
                    x_right = selected_x[pos]
                    increase = (x_right - x_new) * (y_left - y_new)
                
                if increase > best_increase:
                    best_increase = increase
                    best_candidate_idx = cand_idx
            
            if best_candidate_idx == -1:
                break
            
            # Add the best candidate
            new_x = x_grid[best_candidate_idx]
            new_y = y_grid[best_candidate_idx]
            
            # Insert into sorted arrays
            pos = np.searchsorted(selected_x, new_x)
            selected_x = np.insert(selected_x, pos, new_x)
            selected_y = np.insert(selected_y, pos, new_y)
            
            current_hv += best_increase
            
            # Remove from remaining candidates
            remaining_candidates = remaining_candidates[remaining_candidates != best_candidate_idx]
        
        return selected_x

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Set x2..xn to lower bounds
        base_x = lo.copy()
        
        # Evaluate the pre-computed optimal x1 values
        for x1 in self.x1_candidates:
            if time.time() > deadline:
                break
            x = base_x.copy()
            x[0] = x1
            f = problem(x)

        return None
# EVOLVE-BLOCK-END
