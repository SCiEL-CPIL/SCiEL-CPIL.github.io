"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root finding using a global bracketing strategy (bisection).
    This avoids the fragility of Newton's method in ill-conditioned regimes
    where derivatives are noisy or zero.
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
            return f(x)
        except Exception:
            return None

    # Initial evaluation
    x = float(x0)
    fx = safe_f(x)
    if fx is None:
        return x
    if abs(fx) <= 1e-8:
        return x

    # State for bisection
    lo, hi = None, None
    f_lo, f_hi = None, None
    best_x, best_fx = x, fx

    # Helper to update best
    def update_best(x_val, f_val):
        nonlocal best_x, best_fx
        if f_val is not None and abs(f_val) < abs(best_fx):
            best_x, best_fx = x_val, f_val

    # Phase 1: Find a bracket [lo, hi] where f(lo) and f(hi) have opposite signs
    # Start with a small step and expand exponentially
    step = 1e-3 * max(1.0, abs(x))
    # Try expanding in both directions
    for _ in range(50):
        if time.time() > deadline or eval_count >= max_evals:
            break
        
        # Check right
        x_right = x + step
        f_right = safe_f(x_right)
        if f_right is None:
            break
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
            break
        update_best(x_left, f_left)
        if abs(f_left) <= 1e-8:
            return x_left
        if fx * f_left < 0:
            lo, hi = x_left, x
            f_lo, f_hi = f_left, fx
            break
        
        # Expand step
        step *= 2.0
        if step > 1e10:
            break

    # Phase 2: Bisection to refine the root
    if lo is not None and hi is not None:
        for _ in range(100):
            if time.time() > deadline or eval_count >= max_evals:
                break
            
            mid = (lo + hi) / 2.0
            f_mid = safe_f(mid)
            if f_mid is None:
                break
            update_best(mid, f_mid)
            
            if abs(f_mid) <= 1e-8:
                return mid
            
            # Check convergence of interval
            if hi - lo < 1e-14 * max(1.0, abs(mid)):
                return mid
            
            if f_lo * f_mid <= 0:
                hi = mid
                f_hi = f_mid
            else:
                lo = mid
                f_lo = f_mid
        return (lo + hi) / 2.0
    
    # Fallback: If no bracket found, return the best point found
    return best_x
# EVOLVE-BLOCK-END
