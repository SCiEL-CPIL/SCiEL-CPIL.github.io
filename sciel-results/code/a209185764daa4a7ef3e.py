import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        B = self.budget
        calls = 0
        best_f = np.inf
        best_x = np.zeros(n)
        
        def eval_point(x):
            nonlocal calls, best_f, best_x
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Coarse global search using Chebyshev nodes
        # Use a moderate number of points to cover the space
        n_cheb = 12
        # Chebyshev nodes in [-1, 1]
        cheb_nodes = np.cos(np.pi * (np.arange(n_cheb) + 0.5) / n_cheb)
        # Map to [-5, 5]
        cheb_nodes = -5 + 10 * (cheb_nodes + 1) / 2
        cheb_nodes = np.sort(cheb_nodes)  # Sort for consistency

        # Evaluate a grid of points (limited by budget)
        # For n=5, full grid would be 12^5 = 248832, too many
        # Use a sparse grid: evaluate points where each coordinate is at a Chebyshev node
        # but limit total evaluations
        max_grid_evals = min(B // 3, 500)  # Use at most 1/3 of budget or 500
        
        # Generate candidate points using Latin Hypercube-like sampling with Chebyshev nodes
        rng = np.random.RandomState(42)
        candidates = []
        
        # First, evaluate some random points to get an initial sense
        n_random = min(20, max_grid_evals // 5)
        for _ in range(n_random):
            x = rng.uniform(-5, 5, n)
            eval_point(x)
        
        # Then do coordinate-wise exploration
        # For each dimension, try different values while keeping others at best known
        remaining_budget = B - calls
        
        # Initialize with best point found so far
        current_best = best_x.copy()
        
        # Adaptive coordinate descent with bisection
        # For each dimension, perform a 1D search using the function's step-like nature
        
        # Phase 2: Coordinate descent with bisection
        # For each dimension, we'll do a bisection search to find the step boundary
        
        def bisection_search_1d(dim_idx, center, lo, hi, max_evals):
            """
            Perform bisection search along dimension dim_idx.
            center: the current best point
            lo, hi: bounds for the search
            max_evals: maximum number of evaluations
            """
            nonlocal calls, best_f, best_x
            if calls >= B or max_evals <= 0:
                return center[dim_idx], 0
            
            evals_used = 0
            a = lo
            b = hi
            c = center[dim_idx]
            
            # First, evaluate at center to have a reference
            f_center = best_f  # We already know this
            
            # Do bisection: try points and narrow down
            # For step functions, we want to find where the function changes
            # Strategy: sample a few points, then bisect toward the better region
            
            # Sample points: center, and points offset in both directions
            offsets = [0.0, -1.0, 1.0, -0.5, 0.5, -2.0, 2.0, -0.25, 0.25, -0.1, 0.1]
            
            best_val = f_center
            best_offset = 0.0
            
            for offset in offsets:
                if calls >= B or evals_used >= max_evals:
                    break
                test_val = center[dim_idx] + offset
                if test_val < lo or test_val > hi:
                    continue
                x_test = center.copy()
                x_test[dim_idx] = test_val
                f_test = eval_point(x_test)
                evals_used += 1
                if f_test < best_val:
                    best_val = f_test
                    best_offset = offset
            
            # Now do bisection around the best offset found
            if best_offset != 0.0 and calls < B and evals_used < max_evals:
                # Bisect between 0 and best_offset
                left = 0.0
                right = best_offset
                f_left = f_center
                f_right = best_val
                
                for _ in range(5):  # 5 bisection steps
                    if calls >= B or evals_used >= max_evals:
                        break
                    mid = (left + right) / 2
                    x_mid = center.copy()
                    x_mid[dim_idx] = center[dim_idx] + mid
                    f_mid = eval_point(x_mid)
                    evals_used += 1
                    
                    if f_mid <= f_left:
                        left = mid
                        f_left = f_mid
                    else:
                        right = mid
                        f_right = f_mid
                
                # Take the better of left and right
                if f_left <= f_right:
                    best_offset = left
                else:
                    best_offset = right
            
            return center[dim_idx] + best_offset, evals_used

        # Perform coordinate descent iterations
        n_cd_iterations = 3
        for it in range(n_cd_iterations):
            if calls >= B:
                break
            
            # For each dimension, do a bisection search
            for dim_idx in range(n):
                if calls >= B:
                    break
                
                # Determine search bounds: around current best
                center = best_x.copy()
                lo = max(-5, center[dim_idx] - 3)
                hi = min(5, center[dim_idx] + 3)
                
                # Allocate evaluations: more for early iterations
                if it == 0:
                    max_evals_dim = min(30, (B - calls) // n)
                elif it == 1:
                    max_evals_dim = min(20, (B - calls) // n)
                else:
                    max_evals_dim = min(10, (B - calls) // n)
                
                new_val, evals_used = bisection_search_1d(dim_idx, center, lo, hi, max_evals_dim)
                if evals_used > 0:
                    # Update the best point if we found something better
                    x_new = best_x.copy()
                    x_new[dim_idx] = new_val
                    # Check if this is actually better
                    if calls < B:
                        f_new = eval_point(x_new)
                        # best_f and best_x already updated in eval_point
        
        # Phase 3: Fine-tuning with local search around best point
        if calls < B:
            # Do a finer local search
            local_radius = 0.5
            n_local = 10
            for dim_idx in range(n):
                if calls >= B:
                    break
                # Try small perturbations
                for delta in np.linspace(-local_radius, local_radius, n_local):
                    if calls >= B:
                        break
                    x_test = best_x.copy()
                    x_test[dim_idx] = best_x[dim_idx] + delta
                    eval_point(x_test)
            
            # Another pass with even finer resolution
            if calls < B:
                local_radius = 0.1
                n_local = 5
                for dim_idx in range(n):
                    if calls >= B:
                        break
                    for delta in np.linspace(-local_radius, local_radius, n_local):
                        if calls >= B:
                            break
                        x_test = best_x.copy()
                        x_test[dim_idx] = best_x[dim_idx] + delta
                        eval_point(x_test)

        return best_x, best_f
