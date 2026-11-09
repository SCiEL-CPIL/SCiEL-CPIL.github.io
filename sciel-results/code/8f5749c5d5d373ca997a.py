"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import brentq

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    eval_count = 0
    max_evals = 500
    
    def safe_f(x):
        nonlocal eval_count
        if eval_count >= max_evals:
            return None
        eval_count += 1
        try:
            val = f(x)
            if not np.isfinite(val):
                return None
            return float(val)
        except:
            return None

    # Initialize
    x0 = float(x0)
    fx0 = safe_f(x0)
    if fx0 is None:
        return x0
    if abs(fx0) <= 1e-8:
        return x0

    best_x = x0
    best_f = fx0
    best_abs_f = abs(fx0)

    # Phase 1: Robust Secant Method with Damping
    # Use two points to start secant
    h = 1e-4 * max(1.0, abs(x0))
    x1 = x0 + h
    fx1 = safe_f(x1)
    
    if fx1 is None:
        # Fallback to central difference for initial slope if secant start fails
        x1 = x0 - h
        fx1 = safe_f(x1)
        
    if fx1 is None:
        # Can't even get a second point, try Newton with central diff
        x_m = x0 - h
        x_p = x0 + h
        fm = safe_f(x_m)
        fp = safe_f(x_p)
        if fm is not None and fp is not None:
            d = (fp - fm) / (2*h)
            if d != 0:
                x1 = x0 - fx0 / d
                fx1 = safe_f(x1)
            else:
                fx1 = None
        if fx1 is None:
            return best_x

    # Track history for secant
    x_prev = x0
    fx_prev = fx0
    x_curr = x1
    fx_curr = fx1

    for i in range(80):
        if time.time() > deadline:
            break
        if fx_curr is None:
            break
        if abs(fx_curr) < abs(best_f):
            best_x = x_curr
            best_f = fx_curr
            best_abs_f = abs(fx_curr)
        if abs(fx_curr) <= 1e-8:
            return x_curr

        # Secant step
        if fx_curr == fx_prev:
            # Derivative is zero or parallel, perturb
            x_curr = x_curr + 1e-4
            fx_curr = safe_f(x_curr)
            continue
            
        x_next = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
        
        # Damping: if step is too large or doesn't improve, damp it
        step = x_next - x_curr
        max_step = 0.5 * abs(x_curr) + 1.0
        if abs(step) > max_step:
            x_next = x_curr + max_step * math.copysign(1, step)
            
        fx_next = safe_f(x_next)
        
        if fx_next is None:
            # Try smaller step
            x_next = x_curr + 0.5 * step
            fx_next = safe_f(x_next)
            if fx_next is None:
                break

        # Check for improvement
        if abs(fx_next) < abs(fx_curr):
            x_prev = x_curr
            fx_prev = fx_curr
            x_curr = x_next
            fx_curr = fx_next
        else:
            # Damp the step
            x_damped = x_curr + 0.5 * (x_next - x_curr)
            fx_damped = safe_f(x_damped)
            if fx_damped is not None and abs(fx_damped) < abs(fx_curr):
                x_prev = x_curr
                fx_prev = fx_curr
                x_curr = x_damped
                fx_curr = fx_damped
            else:
                # If no improvement, break to try bracketing
                break

    if abs(best_f) <= 1e-8:
        return best_x

    # Phase 2: Bracketing
    # Try to find a bracket [a, b] with sign change
    # Start around best_x
    a = best_x
    fa = safe_f(a)
    if fa is None:
        return best_x
        
    # Expand interval
    r = 1e-4 * max(1.0, abs(a))
    found_bracket = False
    b = a
    fb = None
    
    for i in range(30):
        if time.time() > deadline:
            break
        if eval_count >= max_evals - 10:
            break
            
        # Try expanding to the right
        b = a + r
        fb = safe_f(b)
        if fb is not None:
            if fa * fb < 0:
                found_bracket = True
                break
            # If fa is very small, we might have missed the root, but we check best_x already
            # Continue expanding
        # Try expanding to the left
        b = a - r
        fb = safe_f(b)
        if fb is not None:
            if fa * fb < 0:
                found_bracket = True
                break
        
        r *= 2.0

    if found_bracket:
        # Phase 3: Brent's Method
        try:
            root = brentq(safe_f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
            if root is not None:
                fr = safe_f(root)
                if fr is not None and abs(fr) < abs(best_f):
                    return root
        except:
            pass

    return best_x
# EVOLVE-BLOCK-END
