"""Root finder for 3 fixed oscillatory equations."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using a robust bracketing + secant/bisection
    hybrid approach with a time deadline guard.
    """
    tol = 1e-8
    deadline = time.time() + 110.0  # safety margin before 120s wall-clock limit
    
    x = float(x0)
    
    # Evaluate at starting point
    try:
        fx = f(x)
    except Exception:
        fx = float('nan')
    
    if not math.isfinite(fx):
        # Try a few nearby points
        for dx in [1e-6, -1e-6, 1e-4, -1e-4, 1.0, -1.0]:
            try:
                x_try = x + dx
                fx_try = f(x_try)
                if math.isfinite(fx_try) and abs(fx_try) <= tol:
                    return x_try
            except Exception:
                continue
        return x  # best effort
    
    if abs(fx) <= tol:
        return x
    
    # Phase 1: Find a bracket [a, b] where f(a) and f(b) have opposite signs
    a = x
    fa = fx
    
    # Try expanding brackets from x0
    # Use a sequence of step sizes: small to large
    steps = [1e-8, -1e-8, 1e-6, -1e-6, 1e-4, -1e-4, 1e-2, -1e-2, 
             0.1, -0.1, 1.0, -1.0, 10.0, -10.0, 100.0, -100.0,
             1000.0, -1000.0, 1e4, -1e4, 1e5, -1e5, 1e6, -1e6]
    
    b = None
    fb = None
    
    for delta in steps:
        if time.time() > deadline:
            break
        x_try = x + delta
        try:
            f_try = f(x_try)
        except Exception:
            continue
        
        if not math.isfinite(f_try):
            continue
        
        if abs(f_try) <= tol:
            return x_try
        
        if fa * f_try < 0:
            b = x_try
            fb = f_try
            break
    
    if b is None:
        # No bracket found with simple expansion.
        # Try a more systematic search: sample on a grid around x0
        # Also try to find local minima of |f|
        best_x = x
        best_abs_fx = abs(fx)
        
        # Sample on a logarithmic grid
        for scale in [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5]:
            if time.time() > deadline:
                break
            for sign in [1, -1]:
                x_try = x + sign * scale
                try:
                    f_try = f(x_try)
                except Exception:
                    continue
                if not math.isfinite(f_try):
                    continue
                if abs(f_try) < best_abs_fx:
                    best_abs_fx = abs(f_try)
                    best_x = x_try
                if abs(f_try) <= tol:
                    return x_try
        
        # Try to bracket using the best point found
        if best_abs_fx < abs(fx):
            a = best_x
            fa = f(best_x)
            if abs(fa) <= tol:
                return a
        
        # If still no bracket, try secant method as fallback
        # Use two points: x0 and the best point found
        x1 = best_x
        f1 = fa if abs(fa) < abs(fx) else fx
        x0_sec = x
        f0_sec = fx
        
        # If we have two distinct points with different function values, try secant
        if x1 != x0_sec and f1 != f0_sec:
            for _ in range(100):
                if time.time() > deadline:
                    break
                # Secant step
                if f1 == f0_sec:
                    break
                x_new = x1 - f1 * (x1 - x0_sec) / (f1 - f0_sec)
                try:
                    f_new = f(x_new)
                except Exception:
                    break
                if not math.isfinite(f_new):
                    break
                if abs(f_new) <= tol:
                    return x_new
                # Shift points
                x0_sec = x1
                f0_sec = f1
                x1 = x_new
                f1 = f_new
                if abs(x1 - x0_sec) < 1e-15 * max(1.0, abs(x1)):
                    break
        
        return best_x
    
    # Phase 2: We have a bracket [a, b] with f(a)*f(b) < 0
    # Use a hybrid secant-with-bisection approach
    # Track the two most recent points for secant
    
    # Initialize with a and b
    x_lo = a
    x_hi = b
    f_lo = fa
    f_hi = fb
    
    # For secant, we need two points. Start with a and b.
    x_prev = a
    f_prev = fa
    x_curr = b
    f_curr = fb
    
    for _ in range(200):
        if time.time() > deadline:
            break
        
        # Try secant step
        if f_curr != f_prev:
            x_secant = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        else:
            x_secant = None
        
        # Check if secant step is within bracket
        use_secant = (x_secant is not None and 
                      (x_lo < x_secant < x_hi) and
                      math.isfinite(x_secant))
        
        if use_secant:
            x_new = x_secant
        else:
            # Bisection
            x_new = (x_lo + x_hi) / 2.0
        
        try:
            f_new = f(x_new)
        except Exception:
            # Fall back to bisection
            x_new = (x_lo + x_hi) / 2.0
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break
        
        if not math.isfinite(f_new):
            x_new = (x_lo + x_hi) / 2.0
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break
        
        if abs(f_new) <= tol:
            return x_new
        
        # Update bracket
        if f_lo * f_new < 0:
            x_hi = x_new
            f_hi = f_new
        else:
            x_lo = x_new
            f_lo = f_new
        
        # Update secant points
        x_prev = x_curr
        f_prev = f_curr
        x_curr = x_new
        f_curr = f_new
        
        # Check convergence
        if abs(x_hi - x_lo) < 1e-15 * max(1.0, abs(x_new)):
            break
    
    # Return the point with smallest |f|
    candidates = [x_lo, x_hi, x_curr]
    best_x = x_curr
    best_val = abs(f_curr)
    for cx in candidates:
        if cx == x_curr:
            continue
        try:
            cv = abs(f(cx))
            if math.isfinite(cv) and cv < best_val:
                best_val = cv
                best_x = cx
        except Exception:
            continue
    
    return best_x

# EVOLVE-BLOCK-END
