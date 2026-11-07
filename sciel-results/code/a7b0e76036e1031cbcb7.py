"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Coordinate-wise ladder search with golden-section refinement for Rastrigin optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        calls_made = 0
        dim = self.dim
        budget = self.budget

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        best_x = np.zeros(dim)
        best_f = np.inf

        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse Grid Search to find a good starting region
        # We use a 2^dim grid over [-5, 5]^dim
        n_coarse = 2 ** dim
        if n_coarse <= budget:
            for i in range(n_coarse):
                if calls_made >= budget or time.time() > deadline - 30:
                    break
                coords = []
                idx = i
                for d in range(dim):
                    bit = (idx >> (dim - 1 - d)) & 1
                    coord = -5.0 + bit * 5.0
                    coords.append(coord)
                x = np.array(coords)
                counted_func(x)

        # Phase 2: Coordinate-wise Ladder Search
        # We cycle through dimensions, performing golden-section search on each.
        # This exploits the separability of Rastrigin.
        
        # We will do multiple passes. In each pass, we refine each dimension.
        # Golden section search requires 2 initial points, then 1 per iteration.
        # Let's define a helper for golden section search on a single dimension.
        
        def golden_section_search(func, x_fixed, dim_idx, a, b, max_iter, tol=1e-6):
            """
            Minimizes func(x) with respect to x[dim_idx], keeping other dims fixed at x_fixed.
            a, b are the bounds for x[dim_idx].
            Returns the best x_found for that dimension.
            """
            nonlocal calls_made, best_x, best_f
            
            if calls_made >= budget:
                return x_fixed[dim_idx]
                
            # Golden ratio
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            # Evaluate c and d
            x_c = x_fixed.copy()
            x_c[dim_idx] = c
            x_d = x_fixed.copy()
            x_d[dim_idx] = d
            
            f_c = counted_func(x_c)
            f_d = counted_func(x_d)
            
            if calls_made >= budget:
                # Return the better of the two if we ran out of budget
                if f_c < f_d:
                    return c
                else:
                    return d

            for _ in range(max_iter):
                if time.time() > deadline - 30:
                    break
                if calls_made >= budget:
                    break
                if (b - a) < tol:
                    break
                    
                if f_c < f_d:
                    b = d
                    d = c
                    f_d = f_c
                    c = b - gr * (b - a)
                    x_c[dim_idx] = c
                    f_c = counted_func(x_c)
                else:
                    a = c
                    c = d
                    f_c = f_d
                    d = a + gr * (b - a)
                    x_d[dim_idx] = d
                    f_d = counted_func(x_d)
                    
            # Determine the best point in the final interval
            # We have evaluated c and d. The minimum is at the one with lower f, 
            # or potentially the midpoint if we want to be safe, but usually the 
            # retained point is sufficient.
            if f_c < f_d:
                return c
            else:
                return d

        # We will perform several rounds of ladder search.
        # Start with the best point found in Phase 1.
        # We need to define search ranges. Initially [-5, 5].
        # As we refine, we can narrow the range around the best found so far.
        
        # Let's do 3 passes of ladder search.
        # Pass 1: Full range [-5, 5] for each dim.
        # Pass 2: Narrowed range around the best found.
        # Pass 3: Even narrower.
        
        passes = 3
        current_x = best_x.copy()
        
        for p in range(passes):
            if time.time() > deadline - 30:
                break
            if calls_made >= budget:
                break
            
            # Determine search range for this pass
            if p == 0:
                ranges = [(-5.0, 5.0) for _ in range(dim)]
                # For the first pass, we can use fewer iterations per dim to save budget
                # for the other dims.
                # Total budget ~500. Coarse used 32. Left ~468.
                # 5 dims * (2 + iter) calls per dim.
                # If iter=10, that's 12 calls/dim * 5 = 60 calls/pass.
                # We can afford many passes.
                max_iter_per_dim = 10
            else:
                # Narrow around current best
                # Let's estimate a local box size
                # If we haven't converged, use a fixed small size, else adapt
                # For simplicity, use a fixed shrinkage or a heuristic.
                # Let's use a box of size 1.0 around the current best for pass 1, 0.1 for pass 2.
                # Actually, let's just use the range from the previous best found in that dim.
                # But we don't track per-dim history easily here.
                # Let's just use a fixed small window around the global best.
                window_size = 2.0 / (p + 1) # 2.0, 1.0, 0.66...
                ranges = []
                for d in range(dim):
                    center = current_x[d]
                    a = max(-5.0, center - window_size)
                    b = min(5.0, center + window_size)
                    ranges.append((a, b))
                max_iter_per_dim = 10 # Can afford more now

            # Cycle through dimensions
            for d in range(dim):
                if time.time() > deadline - 30:
                    break
                if calls_made >= budget:
                    break
                
                a, b = ranges[d]
                # Perform golden section search on dimension d
                best_val = golden_section_search(func, current_x, d, a, b, max_iter_per_dim)
                current_x[d] = best_val
                
                # Update global best if this improved it
                # counted_func already updates best_x/best_f if it finds a better point
                # But golden_section_search might return a point that wasn't the absolute best 
                # evaluated inside it if the last eval was worse? No, it returns the best of c,d.
                # However, best_x is updated in counted_func. So best_x is always the global best found so far.
                # We should sync current_x with best_x if best_x is better?
                # Actually, the ladder method works by fixing other dims. If we move to a better 
                # point in dim d, the other dims might not be optimal for this new point.
                # But that's fine, the next cycle will re-optimize them.
                
                # Important: If best_x is significantly better than current_x, 
                # we might want to reset current_x to best_x to not lose the good point?
                # No, the ladder approach relies on the trajectory.
                # However, if best_x found by a previous dim optimization in this pass 
                # is better, we should probably use it as the base for the next dim.
                # Let's just keep using current_x as the base, but if best_f is lower than 
                # the current_x's f value (which we don't track directly), it's tricky.
                
                # Simpler: Just use best_x as the starting point for the next dimension?
                # If we do that, we are doing a steepest descent-like coordinate search.
                # Let's try: after optimizing dim d, set current_x = best_x.copy()
                # This ensures we always start from the best known point.
                current_x = best_x.copy()

        # Phase 3: Final Nelder-Mead Polish
        # Use the remaining budget for a local polish
        if calls_made < budget and time.time() < deadline - 5:
            from scipy.optimize import minimize
            try:
                minimize(
                    counted_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
