"""Root finding for ill-conditioned problems."""
# EVOLVE-BLOCK-START
import time

def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Returns x with |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a combination of:
    1. Newton's method with forward-difference derivative (1 extra eval per step)
    2. Bisection fallback when Newton diverges or no bracket
    3. Bracket expansion to find sign change
    4. Deadline guard for safety
    """
    x0 = float(x0)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    eval_count = 0
    max_evals = 500
    
    def eval_f(x):
        nonlocal eval_count
        if eval_count >= max_evals:
            return None
        eval_count += 1
        try:
            val = f(x)
            return val
        except Exception:
            return None
    
    # Check if x0 is already a root
    fx0 = eval_f(x0)
    if fx0 is not None and abs(fx0) <= 1e-8:
        return x0
    
    # Try Newton's method first
    x = x0
    fx = fx0
    
    # Newton phase with forward difference
    for i in range(150):
        if time.time() > deadline:
            break
        if fx is None or abs(fx) <= 1e-8:
            break
        if eval_count >= max_evals:
            break
        
        # Forward difference derivative
        h = 1e-8 * max(1.0, abs(x))
        fx_plus = eval_f(x + h)
        if fx_plus is None:
            break
        
        d = (fx_plus - fx) / h
        if d == 0 or not _is_finite(d):
            break
        
        x_new = x - fx / d
        
        # Check if step is reasonable
        if not _is_finite(x_new):
            break
        
        # Damping if step is too large
        if abs(x_new - x) > 10.0 * max(1.0, abs(x)):
            x_new = x + 10.0 * max(1.0, abs(x)) * (1.0 if x_new > x else -1.0)
        
        fx_new = eval_f(x_new)
        if fx_new is None:
            break
        
        # Accept step if it reduces |f|
        if abs(fx_new) < abs(fx):
            x = x_new
            fx = fx_new
            if abs(fx) <= 1e-8:
                return x
        else:
            # Try bisection-like step
            # Keep x as is, try smaller step
            x_new = x - 0.5 * fx / d
            if not _is_finite(x_new):
                break
            fx_new = eval_f(x_new)
            if fx_new is None:
                break
            if abs(fx_new) < abs(fx):
                x = x_new
                fx = fx_new
                if abs(fx) <= 1e-8:
                    return x
            # If neither works, break to bisection
            if eval_count >= max_evals:
                break
    
    if fx is not None and abs(fx) <= 1e-8:
        return x
    
    # If Newton didn't converge, try to find a bracket and use bisection
    # First, expand around x to find a sign change
    a = x
    fa = fx if fx is not None else eval_f(x)
    if fa is None:
        # Try x0 again
        a = x0
        fa = eval_f(a)
        if fa is None:
            return x0
    
    # Expand to find bracket
    b = a
    fb = fa
    scale = 1.0
    for i in range(50):
        if time.time() > deadline:
            break
        if eval_count >= max_evals:
            break
        
        # Try expanding in both directions
        if i % 2 == 0:
            b = a + scale
        else:
            b = a - scale
        
        fb = eval_f(b)
        if fb is None:
            break
        
        if fa * fb < 0:
            # Found bracket [a, b]
            break
        
        scale *= 2.0
        if scale > 1e10:
            break
    else:
        # No bracket found, return best x
        return x
    
    # Bisection with Illinois modification
    if fa * fb >= 0:
        return x
    
    # Ensure fa < fb in terms of sign (not value)
    # We just need fa and fb to have opposite signs
    for i in range(100):
        if time.time() > deadline:
            break
        if eval_count >= max_evals:
            break
        
        mid = 0.5 * (a + b)
        fm = eval_f(mid)
        if fm is None:
            break
        
        if abs(fm) <= 1e-8:
            return mid
        
        if fa * fm < 0:
            b = mid
            fb = fm
        else:
            a = mid
            fa = fm
        
        # Check convergence
        if abs(b - a) < 1e-12 * max(1.0, abs(mid)):
            break
    
    # Return the best point
    # Evaluate both endpoints and midpoint to find best
    candidates = [a, b, 0.5*(a+b)]
    best_x = x
    best_val = abs(fx) if fx is not None else float('inf')
    
    for cx in candidates:
        if time.time() > deadline:
            break
        if eval_count >= max_evals:
            break
        cf = eval_f(cx)
        if cf is not None and abs(cf) < best_val:
            best_val = abs(cf)
            best_x = cx
    
    return best_x

def _is_finite(x):
    """Check if x is finite."""
    if x is None:
        return False
    try:
        return abs(x) != float('inf') and not (x != x)  # not NaN
    except:
        return False
# EVOLVE-BLOCK-END
