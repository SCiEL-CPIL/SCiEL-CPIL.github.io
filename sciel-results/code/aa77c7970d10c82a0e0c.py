import math
import time

def find_root(f, x0):
    TOL = 1e-8
    MAX_CALLS = 490
    deadline = time.time() + 145.0
    
    calls = 0
    best_x = float(x0)
    best_abs = float('inf')
    
    def safe_f(x):
        nonlocal calls, best_x, best_abs
        if calls >= MAX_CALLS or time.time() > deadline:
            return float('inf')
        calls += 1
        try:
            val = f(x)
            if not math.isfinite(val):
                return float('inf')
            a = abs(val)
            if a < best_abs:
                best_abs = a
                best_x = x
            return val
        except Exception:
            return float('inf')
    
    x0 = float(x0)
    
    # First, evaluate at x0
    fx0 = safe_f(x0)
    if abs(fx0) <= TOL:
        return x0
    if not math.isfinite(fx0):
        # x0 is in a bad region, try nearby
        for dx in [0.1, -0.1, 0.5, -0.5, 1.0, -1.0]:
            x_try = x0 + dx
            fx_try = safe_f(x_try)
            if abs(fx_try) <= TOL:
                return x_try
            if math.isfinite(fx_try):
                best_x = x_try
                best_abs = abs(fx_try)
                break
        return best_x
    
    # Try to find a bracket near x0 using small steps
    # For oscillatory functions, the period is likely O(1) or O(10)
    # Try multiple scales of local search
    
    bracket_a = None
    bracket_b = None
    
    # Try progressively larger local windows
    for scale in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]:
        if calls >= MAX_CALLS - 50 or time.time() > deadline:
            break
        
        # Check left and right of x0
        xl = x0 - scale
        xr = x0 + scale
        
        fxl = safe_f(xl)
        fxr = safe_f(xr)
        
        if not math.isfinite(fxl) or not math.isfinite(fxr):
            continue
        
        # Check if sign change between xl and x0
        if fx0 * fxl < 0:
            bracket_a = xl
            bracket_b = x0
            break
        
        # Check if sign change between x0 and xr
        if fx0 * fxr < 0:
            bracket_a = x0
            bracket_b = xr
            break
        
        # Check if sign change between xl and xr (x0 might be outside)
        if fxl * fxr < 0:
            bracket_a = xl
            bracket_b = xr
            break
        
        # Also check midpoint for tighter bracket
        xm = (xl + xr) / 2.0
        fxm = safe_f(xm)
        if math.isfinite(fxm):
            if fx0 * fxm < 0:
                if fx0 > fxm:
                    bracket_a = x0
                    bracket_b = xm
                else:
                    bracket_a = xm
                    bracket_b = x0
                break
            if fxl * fxm < 0:
                bracket_a = xl
                bracket_b = xm
                break
            if fxm * fxr < 0:
                bracket_a = xm
                bracket_b = xr
                break
    
    if bracket_a is not None:
        # Use bisection + IQUO hybrid to refine
        a, b = bracket_a, bracket_b
        fa = safe_f(a)
        fb = safe_f(b)
        
        # Ensure fa and fb have opposite signs
        if fa * fb > 0:
            fa = safe_f(a)
            fb = safe_f(b)
        
        # Track previous point for IQUO
        c_prev = None
        fc_prev = None
        
        for _ in range(80):
            if calls >= MAX_CALLS - 10 or time.time() > deadline:
                break
            if fa * fb > 0:
                break
            if abs(b - a) < 1e-15:
                break
            
            # Try IQUO (inverse quadratic interpolation) if we have 3 points
            xn = None
            if c_prev is not None and fc_prev is not None:
                # Check if IQUO is valid
                if (a != b and b != c_prev and a != c_prev and
                    fa != fb and fb != fc_prev and fa != fc_prev):
                    # Inverse quadratic interpolation
                    # f(x) ≈ L(x) / Q(x) where we interpolate 1/f
                    # Using Lagrange form for 1/f at points a, b, c_prev
                    try:
                        inv_fa = 1.0 / fa
                        inv_fb = 1.0 / fb
                        inv_fc = 1.0 / fc_prev
                        
                        # Quadratic through (a, inv_fa), (b, inv_fb), (c_prev, inv_fc)
                        # Find x where this quadratic = 0
                        # Using Lagrange basis:
                        # P(x) = inv_fa * (x-b)(x-c_prev)/((a-b)(a-c_prev)) 
                        #       + inv_fb * (x-a)(x-c_prev)/((b-a)(b-c_prev))
                        #       + inv_fc * (x-a)(x-b)/((c_prev-a)(c_prev-b))
                        # Set P(x) = 0 and solve quadratic
                        # This is complex; use a simpler secant-based approach instead
                        pass
                    except (ZeroDivisionError, OverflowError):
                        xn = None
            
            # Fall back to secant/Newton step
            if xn is None:
                if fb != fa and abs(b - a) > 1e-15:
                    deriv = (fb - fa) / (b - a)
                else:
                    deriv = 1.0
                
                if abs(deriv) > 1e-15:
                    xn = b - fb / deriv
                else:
                    xn = (a + b) / 2.0
            
            # Keep xn in bracket
            if not (a < xn < b) or not math.isfinite(xn):
                xn = (a + b) / 2.0
            
            fxn = safe_f(xn)
            
            if abs(fxn) <= TOL:
                return xn
            
            if not math.isfinite(fxn):
                xn = (a + b) / 2.0
                fxn = safe_f(xn)
                if not math.isfinite(fxn):
                    break
            
            # Update bracket
            if fa * fxn < 0:
                c_prev = b
                fc_prev = fb
                b = xn
                fb = fxn
            else:
                c_prev = a
                fc_prev = fa
                a = xn
                fa = fxn
        
        # Final polish: use secant method with tight tolerance from best endpoint
        # Start from the endpoint with smaller |f|
        if abs(fa) < abs(fb):
            x_curr = a
            fx_curr = fa
            x_prev = b
            fx_prev = fb
        else:
            x_curr = b
            fx_curr = fb
            x_prev = a
            fx_prev = fa
        
        # Secant polish with tighter tolerance
        POLISH_TOL = 1e-12
        for _ in range(20):
            if calls >= MAX_CALLS - 3 or time.time() > deadline:
                break
            if abs(fx_curr) <= POLISH_TOL:
                return x_curr
            if fx_prev == fx_curr:
                break
            
            x_new = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
            
            if not math.isfinite(x_new):
                break
            
            fx_new = safe_f(x_new)
            
            if abs(fx_new) <= TOL:
                return x_new
            
            if not math.isfinite(fx_new):
                break
            
            x_prev, fx_prev = x_curr, fx_curr
            x_curr, fx_curr = x_new, fx_new
        
        # Return the better endpoint
        if abs(fa) < abs(fb):
            return a
        else:
            return b
    
    # No bracket found - use secant method iteratively
    # This works well for smooth oscillatory functions
    x_prev = x0 - 0.1
    fx_prev = safe_f(x_prev)
    x_curr = x0
    fx_curr = safe_f(x_curr)
    
    if abs(fx_curr) <= TOL:
        return x_curr
    
    for _ in range(80):
        if calls >= MAX_CALLS - 5 or time.time() > deadline:
            break
        
        if abs(fx_prev) < abs(fx_curr):
            # Swap to keep convergence
            x_prev, x_curr = x_curr, x_prev
            fx_prev, fx_curr = fx_curr, fx_prev
        
        if abs(fx_curr) <= TOL:
            return x_curr
        
        if fx_prev == fx_curr:
            break
        
        x_new = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
        
        # Damp if step is too large
        if abs(x_new - x_curr) > 10.0 * max(1.0, abs(x_curr)):
            x_new = x_curr - 10.0 * max(1.0, abs(x_curr)) * (1.0 if fx_curr > 0 else -1.0)
        
        fx_new = safe_f(x_new)
        
        if abs(fx_new) <= TOL:
            return x_new
        
        if not math.isfinite(fx_new):
            # Try smaller step
            x_new = x_curr - 0.1 * (x_curr - x_prev)
            fx_new = safe_f(x_new)
            if not math.isfinite(fx_new):
                break
        
        x_prev, fx_prev = x_curr, fx_curr
        x_curr, fx_curr = x_new, fx_new
    
    # Return best found
    return best_x
