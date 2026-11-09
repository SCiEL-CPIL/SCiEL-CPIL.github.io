"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start L-BFGS-B optimizer with refined coordinate-wise polish for BBOB-style f19."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, aim for 280s to be safe
        deadline = start_time + 280.0
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        call_count = 0
        best_x = None
        best_f = np.inf
        bounds = [(-5, 5)] * n

        def counted_func(x):
            nonlocal call_count
            if call_count >= budget:
                raise ValueError("Budget exceeded")
            call_count += 1
            return func(x)

        # 1. Generate high-quality starting points
        # Strategy: Origin + Stratified (LHS-like) points to cover the domain efficiently
        starts = []
        # Always include origin, as it's a common reference point
        starts.append(np.zeros(n))
        
        # Add 7 stratified points (Total 8 starts)
        # This reduces overhead of many bad starts and focuses on coverage
        n_stratified = 7
        for i in range(n_stratified):
            # Stratified sampling: divide range into bins, pick random point in each
            # This is a simple LHS approximation
            x = np.array([(i + rng.random()) / (n_stratified + 1) * 10 - 5 for _ in range(n)])
            starts.append(x)

        # 2. Multi-start L-BFGS-B
        # We have 500 calls. 
        # L-BFGS-B in 5D typically needs ~50-100 calls to converge well.
        # Let's allow up to 60 calls per start.
        # 8 starts * 60 = 480 calls. This leaves ~20 for polish.
        # If a start converges fast, we save calls for others or polish.
        
        max_evals_per_start = 60
        
        for x0 in starts:
            if time.time() > deadline:
                break
            if call_count >= budget:
                break
            
            remaining = budget - call_count
            if remaining < 10:
                break
                
            # Cap the evaluations for this specific run
            current_max_fun = min(remaining, max_evals_per_start)
            
            try:
                res = minimize(
                    counted_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={
                        'maxiter': 100,
                        'maxfun': current_max_fun,
                        'gtol': 1e-8,  # Tighter gradient tolerance
                        'ftol': 1e-12  # Tighter function tolerance
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                # If an exception occurs, we just move to the next start
                pass

        # 3. Final High-Precision Polish: Cyclic Coordinate-wise Golden Section Search
        # The directive suggests using this to exploit local structure and ensure precision.
        # We will iterate through dimensions, performing a golden section search on each.
        # We repeat this cycle until we run out of budget or time.
        
        if best_x is not None:
            x = best_x.copy()
            f_curr = best_f
            
            # Golden ratio constants
            inv_phi = (np.sqrt(5) - 1) / 2
            inv_phi2 = inv_phi * inv_phi
            
            # Adaptive step size for initial bracketing
            # Start with a moderate step
            initial_step = 1.0
            
            # We will perform multiple passes of coordinate descent
            # Each pass iterates through all dimensions
            # We limit the number of iterations per coordinate search to save calls
            
            max_coord_iter = 30  # Max iterations for golden section per coordinate
            max_passes = 10      # Max full passes through all dimensions
            
            for _pass in range(max_passes):
                if call_count >= budget or time.time() > deadline:
                    break
                
                improved = False
                for dim_i in range(n):
                    if call_count >= budget or time.time() > deadline:
                        break
                    
                    # Try to bracket the minimum along dimension dim_i
                    # We start from the current best point x
                    x_center = x.copy()
                    f_center = f_curr
                    
                    # Initial bracket attempt
                    step = initial_step
                    # Shrink step if we are near boundaries or if previous steps failed
                    # Simple adaptive step: try 1.0, then 0.1, then 0.01
                    brackets = [1.0, 0.1, 0.01, 0.001]
                    found_bracket = False
                    a = b = fa = fb = None
                    
                    for s in brackets:
                        x_minus = x_center.copy()
                        x_plus = x_center.copy()
                        x_minus[dim_i] = np.clip(x_center[dim_i] - s, -5, 5)
                        x_plus[dim_i] = np.clip(x_center[dim_i] + s, -5, 5)
                        
                        # Check if we are actually moving
                        if np.allclose(x_minus, x_plus):
                            continue
                            
                        try:
                            f_minus = counted_func(x_minus)
                            f_plus = counted_func(x_plus)
                        except Exception:
                            continue
                            
                        # If center is better than both, we might be at a local min for this coord
                        # Or the step is too large and we jumped over the min? 
                        # Golden section requires f(a) and f(b) to be higher than f(c) or f(d) eventually.
                        # If f_center < f_minus and f_center < f_plus, we are likely at the bottom of this coordinate valley.
                        if f_center <= f_minus and f_center <= f_plus:
                            # Try a smaller step to be sure
                            continue
                        
                        # If one side is better, bracket is [center, better_side]
                        # If both sides are better, bracket is [minus, plus]
                        
                        if f_minus < f_plus:
                            # Min is likely to the left
                            if f_minus < f_center:
                                a, b = x_minus[dim_i], x_center[dim_i]
                                fa, fb = f_minus, f_center
                                found_bracket = True
                            else:
                                # f_center <= f_minus but f_plus < f_center? No, we checked f_center <= both.
                                # If f_minus < f_plus but f_center < f_minus, we are in a weird spot.
                                # Let's assume the bracket is [minus, center] if f_minus < f_center is false?
                                # Actually, if f_center < f_minus and f_center < f_plus, we break.
                                pass
                        else:
                            # Min is likely to the right
                            if f_plus < f_center:
                                a, b = x_center[dim_i], x_plus[dim_i]
                                fa, fb = f_center, f_plus
                                found_bracket = True
                            else:
                                pass
                        
                        if found_bracket:
                            break
                        
                    if not found_bracket:
                        # Could not find a bracket with these steps, skip this dimension for now
                        continue
                        
                    # Perform Golden Section Search within [a, b]
                    # Note: a < b is assumed. We ensured this by construction above.
                    for _iter in range(max_coord_iter):
                        if call_count >= budget or time.time() > deadline:
                            break
                            
                        # Ensure a < b
                        if a > b:
                            a, b = b, a
                            fa, fb = fb, fa
                            
                        # Calculate new points
                        c = b - inv_phi * (b - a)
                        d = a + inv_phi * (b - a)
                        
                        # Ensure c < d
                        if c > d:
                            c, d = d, c
                            
                        x_c = x_center.copy()
                        x_d = x_center.copy()
                        x_c[dim_i] = c
                        x_d[dim_i] = d
                        
                        try:
                            fc = counted_func(x_c)
                            fd = counted_func(x_d)
                        except Exception:
                            break
                            
                        if fc < fd:
                            # Minimum is in [a, d]
                            b = d
                            fb = fd
                            # c becomes the new d in the next iteration? 
                            # Standard golden section reuses one point.
                            # Here we just recalculate. It's simpler and safer for small N.
                        else:
                            # Minimum is in [c, b]
                            a = c
                            fa = fc
                            
                    # After golden section, take the midpoint of the final interval
                    x_best_coord = (a + b) / 2.0
                    x_best_coord = np.clip(x_best_coord, -5, 5)
                    
                    x_new = x_center.copy()
                    x_new[dim_i] = x_best_coord
                    
                    try:
                        f_new = counted_func(x_new)
                    except Exception:
                        f_new = np.inf
                        
                    if f_new < f_curr:
                        x = x_new
                        f_curr = f_new
                        best_x = x.copy()
                        best_f = f_curr
                        improved = True
                    else:
                        # If no improvement, we might have over-shoot or noise.
                        # Keep the old x.
                        pass

                # If no dimension improved in this full pass, we might have converged
                if not improved:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
