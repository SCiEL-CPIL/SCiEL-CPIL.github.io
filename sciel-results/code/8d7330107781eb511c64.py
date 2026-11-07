import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        dim = self.dim
        budget = self.budget
        deadline = start_time + 290.0

        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def elapsed_ok():
            return time.time() < deadline

        def eval_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget or not elapsed_ok():
                return None
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse grid scan to find promising region
        # Use a 4^dim grid (4 points per axis) = 4^5 = 1024 points, too many
        # Use 3^dim = 243 points, or sample strategically
        # Let's do a coarse scan with ~100 points using stratified sampling
        
        scan_points = []
        # Stratified sampling: divide each axis into 3 segments, pick midpoints + corners
        # Actually, let's use a simple approach: evaluate at center and offsets
        # Start with a small set of strategic points
        
        # Evaluate at origin and axis-aligned points
        strategic = [np.zeros(dim)]
        for i in range(dim):
            for offset in [-3.0, -1.5, 1.5, 3.0]:
                pt = np.zeros(dim)
                pt[i] = offset
                strategic.append(pt)
        
        for p in strategic:
            if not elapsed_ok():
                break
            fx = eval_func(p)
            if fx is None:
                break

        if best_x is None:
            return np.zeros(dim), np.inf

        # Phase 2: Coordinate descent with binary search on each axis
        # For each coordinate, try to find the step boundary via bisection
        x = best_x.copy()
        fx = best_f
        
        # Number of bisection iterations per coordinate per pass
        # With budget 500 and ~20 used in phase 1, we have ~480 left
        # dim=5, if we do 2 passes with 10 bisections each: 5*10*2 = 100 evals per pass
        # That leaves plenty for multiple passes
        
        max_bisection_depth = 8  # ~256 subdivisions, precision ~10/256 ≈ 0.04
        num_passes = 3  # Multiple passes to refine
        
        for pass_num in range(num_passes):
            if not elapsed_ok() or calls_made >= budget:
                break
            
            for i in range(dim):
                if not elapsed_ok() or calls_made >= budget:
                    break
                
                # Binary search on coordinate i
                lo = -5.0
                hi = 5.0
                # We know x[i] is currently good; search around it
                # But for step functions, we want to find the edge
                # Try both sides
                
                # First, determine direction: is lower or higher better?
                test_low = x.copy()
                test_low[i] = max(-5.0, x[i] - 2.0)
                test_high = x.copy()
                test_high[i] = min(5.0, x[i] + 2.0)
                
                fx_low = eval_func(test_low)
                if fx_low is None:
                    break
                fx_high = eval_func(test_high)
                if fx_high is None:
                    break
                
                # Determine which direction to bisect
                if fx_low < fx_high:
                    # Lower is better, bisect in [test_low[i], x[i]]
                    search_lo = test_low[i]
                    search_hi = x[i]
                elif fx_high < fx_low:
                    # Higher is better, bisect in [x[i], test_high[i]]
                    search_lo = x[i]
                    search_hi = test_high[i]
                else:
                    # Equal, try wider range
                    search_lo = -5.0
                    search_hi = 5.0
                
                # Now bisect to find the step
                for depth in range(max_bisection_depth):
                    if not elapsed_ok() or calls_made >= budget:
                        break
                    
                    mid = (search_lo + search_hi) / 2.0
                    
                    # Evaluate at mid
                    test_mid = x.copy()
                    test_mid[i] = mid
                    fx_mid = eval_func(test_mid)
                    if fx_mid is None:
                        break
                    
                    # Compare with current best at x
                    # If mid is better, the step is likely between mid and the worse side
                    if fx_mid < fx:
                        # mid is better, so the step boundary is between mid and the worse side
                        # Move the search to be closer to mid
                        if fx_mid < fx_low and fx_mid < fx_high:
                            # mid is the best so far
                            x[i] = mid
                            fx = fx_mid
                            best_x = x.copy()
                            best_f = fx
                        # Narrow the search interval
                        if search_lo < mid:
                            # We were searching [lo, hi] and mid is good
                            # The step is between mid and the bad side
                            # Keep the side that contains the improvement
                            if fx_low < fx_high:
                                # Lower was better, so step is between mid and hi
                                search_hi = mid
                            else:
                                # Higher was better, step is between lo and mid
                                search_lo = mid
                        else:
                            search_lo = mid
                    else:
                        # mid is not better, step is on the other side
                        if fx_low < fx_high:
                            # Lower was better, so step is between lo and mid
                            search_hi = mid
                        else:
                            # Higher was better, step is between mid and hi
                            search_lo = mid
        
        return best_x, best_f
