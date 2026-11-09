"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root finding using a hybrid strategy:
    1. Attempt to find a bracket via adaptive expansion.
    2. Use a high-order convergent method (Illinois or Brent-like) within the bracket.
    3. Fallback to global minimization of |f(x)| if no bracket is found.
    
    This approach prioritizes rapid convergence once a sign change is detected,
    while maintaining robustness through the bracketing guarantee.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    eval_count = 0
    max_evals = 500
    
    # Wrapper to count evaluations and handle exceptions
    def safe_f(x):
        nonlocal eval_count
        if eval_count >= max_evals:
            return None
        if time.time() > deadline:
            return None
        eval_count += 1
        try:
            val = f(x)
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                return None
            return float(val)
        except Exception:
            return None

    # Initial evaluation
    x = float(x0)
    fx = safe_f(x)
    if fx is None:
        return x
    if abs(fx) <= 1e-8:
        return x

    # State for best point found so far
    best_x, best_fx = x, fx

    def update_best(x_val, f_val):
        nonlocal best_x, best_fx
        if f_val is not None and abs(f_val) < abs(best_fx):
            best_x, best_fx = x_val, f_val

    # Phase 1: Find a bracket [lo, hi] where f(lo) and f(hi) have opposite signs
    # Use adaptive step size starting small and expanding
    step = 1e-6 * max(1.0, abs(x))
    lo, hi = None, None
    f_lo, f_hi = None, None
    
    # Try expanding in both directions
    for _ in range(40):
        if time.time() > deadline or eval_count >= max_evals:
            break
        
        # Check right
        x_right = x + step
        f_right = safe_f(x_right)
        if f_right is None:
            # If evaluation fails, try a smaller step or break
            pass
        else:
            update_best(x_right, f_right)
            if abs(f_right) <= 1e-8:
                return x_right
            if fx * f_right < 0:
                lo, hi = x, x_right
                f_lo, f_hi = fx, f_right
                break
        
        # Check left
        x_left = x - step
        f_left = safe_f(x_left)
        if f_left is None:
            pass
        else:
            update_best(x_left, f_left)
            if abs(f_left) <= 1e-8:
                return x_left
            if fx * f_left < 0:
                lo, hi = x_left, x
                f_lo, f_hi = f_left, fx
                break
        
        # Expand step exponentially
        step *= 2.0
        if step > 1e12:
            break

    # Phase 2: If bracket found, use Illinois method (modified bisection) for fast convergence
    if lo is not None and hi is not None:
        # Illinois method is often faster than pure bisection for ill-conditioned problems
        # because it maintains a bracket but uses linear interpolation when it converges.
        for _ in range(100):
            if time.time() > deadline or eval_count >= max_evals:
                break
            
            # Linear interpolation (secant step)
            if f_lo != f_hi:
                x_int = (lo * f_hi - hi * f_lo) / (f_hi - f_lo)
            else:
                x_int = (lo + hi) / 2.0
            
            # Ensure x_int is within [lo, hi]
            if not (lo <= x_int <= hi):
                x_int = (lo + hi) / 2.0
            
            f_int = safe_f(x_int)
            if f_int is None:
                break
            update_best(x_int, f_int)
            
            if abs(f_int) <= 1e-8:
                return x_int
            
            # Update bracket
            if f_lo * f_int <= 0:
                hi = x_int
                f_hi = f_int
            else:
                lo = x_int
                f_lo = f_int
            
            # Check convergence
            if hi - lo < 1e-14 * max(1.0, abs(x_int)):
                return x_int
        
        return (lo + hi) / 2.0
    
    # Fallback: If no bracket found, perform a local search around x0
    # This handles cases where the root is very close but no sign change was captured
    # or where the function is always positive/negative near the root (tangent root).
    # We perform a dense local grid search to minimize |f(x)|.
    local_step = 1e-6 * max(1.0, abs(x))
    for _ in range(50):
        if time.time() > deadline or eval_count >= max_evals:
            break
        
        # Check neighbors
        for dx in [-local_step, local_step, -2*local_step, 2*local_step]:
            x_new = x + dx
            f_new = safe_f(x_new)
            if f_new is None:
                continue
            update_best(x_new, f_new)
            if abs(f_new) <= 1e-8:
                return x_new
        
        # Shrink or expand based on best result
        # If best is at a neighbor, move towards it
        if abs(best_fx) < abs(fx):
            # Move towards best_x
            x = best_x
            fx = best_fx
            local_step *= 0.5
        else:
            # Expand search
            local_step *= 2.0
            if local_step > 1e6:
                break
        
        # Periodically re-evaluate center
        if eval_count % 5 == 0:
            f_center = safe_f(x)
            if f_center is not None:
                update_best(x, f_center)
                if abs(f_center) <= 1e-8:
                    return x

    return best_x
# EVOLVE-BLOCK-END
