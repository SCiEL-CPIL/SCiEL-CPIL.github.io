"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START

def _is_finite(x):
    try:
        return math.isfinite(x)
    except (TypeError, ValueError):
        return False

def _safe_f(f, x):
    """Call f(x) safely, returning None on error or non-finite."""
    try:
        val = f(x)
        val = float(val)
        if _is_finite(val):
            return val
    except Exception:
        pass
    return None

def _secant_step(f, x0, x1, max_iter=40, tol=1e-12, deadline=None):
    """
    Secant method. Returns (x, fx, n_calls) or (None, None, n_calls) on failure.
    """
    x_prev = x0
    x_curr = x1
    fx_prev = _safe_f(f, x_prev)
    fx_curr = _safe_f(f, x_curr)
    n_calls = 2
    
    if fx_prev is None or fx_curr is None:
        return None, None, n_calls
    
    if abs(fx_prev) <= tol:
        return x_prev, fx_prev, n_calls
    if abs(fx_curr) <= tol:
        return x_curr, fx_curr, n_calls
    
    for _ in range(max_iter):
        if deadline is not None and time.time() >= deadline:
            # Return best so far
            if abs(fx_prev) < abs(fx_curr):
                return x_prev, fx_prev, n_calls
            else:
                return x_curr, fx_curr, n_calls
        
        if fx_curr == fx_prev:
            # Degenerate: no progress
            break
        
        denom = fx_curr - fx_prev
        if denom == 0:
            break
            
        x_new = x_curr - fx_curr * (x_curr - x_prev) / denom
        
        if not _is_finite(x_new):
            break
            
        fx_new = _safe_f(f, x_new)
        n_calls += 1
        
        if fx_new is None:
            break
            
        if abs(fx_new) <= tol:
            return x_new, fx_new, n_calls
            
        # Shift window
        x_prev, x_curr = x_curr, x_new
        fx_prev, fx_curr = fx_curr, fx_new
        
        # Check for stagnation
        if abs(x_new - x_curr) < 1e-14 * max(1.0, abs(x_new)):
            break
            
    # Return best of last few points
    if abs(fx_prev) < abs(fx_curr):
        return x_prev, fx_prev, n_calls
    else:
        return x_curr, fx_curr, n_calls

def _bisection_step(f, a, b, fa, fb, max_iter=60, tol=1e-12, deadline=None):
    """
    Bisection method. Returns (x, fx, n_calls).
    Assumes fa and fb have opposite signs.
    """
    n_calls = 2  # fa and fb already computed
    
    if abs(fa) <= tol:
        return a, fa, n_calls
    if abs(fb) <= tol:
        return b, fb, n_calls
    
    for _ in range(max_iter):
        if deadline is not None and time.time() >= deadline:
            mid = (a + b) / 2.0
            fm = _safe_f(f, mid)
            n_calls += 1
            if fm is not None:
                return mid, fm, n_calls
            return mid, None, n_calls
            
        mid = (a + b) / 2.0
        fm = _safe_f(f, mid)
        n_calls += 1
        
        if fm is None:
            break
            
        if abs(fm) <= tol:
            return mid, fm, n_calls
            
        # Check for stagnation
        if b - a < 1e-14 * max(1.0, abs(mid)):
            return mid, fm, n_calls
            
        if fa * fm < 0:
            b = mid
            fb = fm
        else:
            a = mid
            fa = fm
            
    mid = (a + b) / 2.0
    fm = _safe_f(f, mid)
    n_calls += 1
    if fm is not None:
        return mid, fm, n_calls
    return mid, None, n_calls

def _find_sign_change(f, x0, max_points=20, deadline=None):
    """
    Try to find a sign change near x0 by expanding interval.
    Returns (a, b, fa, fb, n_calls) or None.
    """
    x = float(x0)
    fx = _safe_f(f, x)
    n_calls = 1
    
    if fx is None:
        return None
    if abs(fx) <= 1e-12:
        return (x, x, fx, fx, n_calls)
    
    sign_f = 1 if fx > 0 else -1
    
    # Try expanding in both directions
    step = 1.0
    for i in range(max_points):
        if deadline is not None and time.time() >= deadline:
            break
            
        # Try x + step
        x_plus = x + step
        fx_plus = _safe_f(f, x_plus)
        n_calls += 1
        if fx_plus is not None:
            if abs(fx_plus) <= 1e-12:
                return (x_plus, x_plus, fx_plus, fx_plus, n_calls)
            sign_plus = 1 if fx_plus > 0 else -1
            if sign_plus != sign_f:
                return (x, x_plus, fx, fx_plus, n_calls)
        
        # Try x - step
        x_minus = x - step
        fx_minus = _safe_f(f, x_minus)
        n_calls += 1
        if fx_minus is not None:
            if abs(fx_minus) <= 1e-12:
                return (x_minus, x_minus, fx_minus, fx_minus, n_calls)
            sign_minus = 1 if fx_minus > 0 else -1
            if sign_minus != sign_f:
                return (x_minus, x, fx_minus, fx, n_calls)
        
        step *= 2.0
        
        # Safety: don't go too far
        if step > 1e6:
            break
    
    return None

def find_root(f, x0):
    """
    Find a root of f near x0 such that |f(x)| <= 1e-8.
    Uses a hybrid approach: secant method with bisection fallback.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    x0 = float(x0)
    
    # First, try secant method with a second point close to x0
    h = 1e-4 * max(1.0, abs(x0))
    x1 = x0 + h
    
    # Get initial values
    fx0 = _safe_f(f, x0)
    fx1 = _safe_f(f, x1)
    n_calls = 2
    
    # If we have a root already
    if fx0 is not None and abs(fx0) <= 1e-12:
        return x0
    if fx1 is not None and abs(fx1) <= 1e-12:
        return x1
    
    # Try secant if both values are finite
    if fx0 is not None and fx1 is not None:
        x, fx, calls = _secant_step(f, x0, x1, max_iter=30, tol=1e-12, deadline=deadline)
        n_calls += calls - 2  # _secant_step counts its own 2 calls
        
        if x is not None and fx is not None and abs(fx) <= 1e-8:
            return x
    
    # If secant didn't work, try to find a sign change and use bisection
    sc = _find_sign_change(f, x0, max_points=15, deadline=deadline)
    if sc is not None:
        a, b, fa, fb, calls_sc = sc
        n_calls += calls_sc
        
        if fa is not None and fb is not None:
            if fa * fb > 0:
                # Same sign, no bracket
                pass
            else:
                x, fx, calls_bis = _bisection_step(f, a, b, fa, fb, max_iter=60, tol=1e-12, deadline=deadline)
                n_calls += calls_bis - 2
                
                if x is not None and fx is not None and abs(fx) <= 1e-8:
                    return x
    
    # Last resort: try a few more secant starts with different h
    for h_factor in [1e-3, 1e-2, 0.1, 1.0, -1e-3, -1e-2, -0.1, -1.0]:
        if time.time() >= deadline:
            break
        h = h_factor * max(1.0, abs(x0))
        x1 = x0 + h
        fx0 = _safe_f(f, x0)
        fx1 = _safe_f(f, x1)
        n_calls += 2
        
        if fx0 is None or fx1 is None:
            continue
        if abs(fx0) <= 1e-12:
            return x0
        if abs(fx1) <= 1e-12:
            return x1
            
        x, fx, calls = _secant_step(f, x0, x1, max_iter=25, tol=1e-12, deadline=deadline)
        n_calls += calls - 2
        
        if x is not None and fx is not None and abs(fx) <= 1e-8:
            return x
            
    # Return the best we found, or x0 as fallback
    # We need to return something finite
    if n_calls > 0:
        # Try to return x0 if f(x0) is finite
        fx0_final = _safe_f(f, x0)
        if fx0_final is not None:
            return x0
        return x0
    
    return x0

# EVOLVE-BLOCK-END
