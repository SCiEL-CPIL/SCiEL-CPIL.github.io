"""Root finder for smooth polynomial equations with call-budget awareness."""
import time
import math

_DEADLINE_START = time.time()
_WALL_LIMIT = 115.0  # safety margin before 120s kill
_SAFETY_MARGIN = 5.0


def _time_left():
    return _WALL_LIMIT - _SAFETY_MARGIN - (time.time() - _DEADLINE_START)


def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using a combination of:
    1. Initial bracketing via adaptive step
    2. Inverse quadratic interpolation (IQUAD) for fast convergence
    3. Bisection fallback for robustness
    4. Newton polish at the end
    """
    x = float(x0)
    
    # Track best solution found so far
    best_x = x
    best_f = f(x)
    best_abs = abs(best_f)
    
    if best_abs <= 1e-8:
        return best_x
    
    # Track call count
    call_count = 1
    
    def eval_f(x_val):
        nonlocal call_count
        if _time_left() < 1.0:
            return None  # signal timeout
        val = f(x_val)
        call_count += 1
        if not _is_finite(val):
            return None
        return float(val)
    
    # Step 1: Try to find a bracket [a, b] where f changes sign
    a = x
    fa = best_f
    
    # Adaptive step size to find bracket
    step = 1.0
    b = None
    fb = None
    
    # Try expanding outward from x0
    for direction in [1, -1]:
        if _time_left() < 1.0:
            break
        for _ in range(20):
            x_try = x + direction * step
            f_try = eval_f(x_try)
            if f_try is None:
                break
            if f_try == 0.0:
                return x_try
            if fa * f_try < 0:
                # Found bracket
                if direction == 1:
                    a, b = x, x_try
                    fa, fb = best_f, f_try
                else:
                    a, b = x_try, x
                    fa, fb = f_try, best_f
                break
            step *= 2.0
            if step > 1e6:
                break
        if b is not None:
            break
    
    # If we found a bracket, use IQUAD/Bisection hybrid
    if b is not None:
        # Inverse Quadratic Interpolation with bisection fallback
        # We need 3 points for IQUAD; start with 2 and use secant for 3rd
        xa, ya = a, fa
        xb, yb = b, fb
        
        # Use secant to get a third point
        if ya == yb:
            # Degenerate case, just bisect
            for _ in range(50):
                if _time_left() < 1.0:
                    break
                xm = (xa + xb) / 2.0
                ym = eval_f(xm)
                if ym is None:
                    break
                if abs(ym) <= 1e-8:
                    return xm
                if ya * ym <= 0:
                    xb, yb = xm, ym
                else:
                    xa, ya = xm, ym
                if abs(xb - xa) < 1e-14:
                    break
            # Return midpoint of final bracket
            return (xa + xb) / 2.0
        
        # Secant step
        denom = yb - ya
        if abs(denom) < 1e-30:
            # Fallback to bisection
            for _ in range(50):
                if _time_left() < 1.0:
                    break
                xm = (xa + xb) / 2.0
                ym = eval_f(xm)
                if ym is None:
                    break
                if abs(ym) <= 1e-8:
                    return xm
                if ya * ym <= 0:
                    xb, yb = xm, ym
                else:
                    xa, ya = xm, ym
                if abs(xb - xa) < 1e-14:
                    break
            return (xa + xb) / 2.0
        
        xc = xb - yb * (xb - xa) / (yb - ya)
        
        # Ensure xc is in bracket
        if not (xa < xc < xb):
            xc = (xa + xb) / 2.0
        
        yc = eval_f(xc)
        if yc is None:
            # Timeout or error, return best
            return best_x
        if abs(yc) <= 1e-8:
            return xc
        
        # Now we have 3 points: (xa, ya), (xb, yb), (xc, yc)
        # Use IQUAD for subsequent iterations
        for _ in range(50):
            if _time_left() < 1.0:
                break
            
            # Update bracket
            if ya * yc <= 0:
                xb, yb = xc, yc
            else:
                xa, ya = xc, yc
            
            # Check if bracket is small enough
            if abs(xb - xa) < 1e-14:
                return (xa + xb) / 2.0
            
            # IQUAD: fit quadratic in x through (xa,ya), (xb,yb), (xc,yc)
            # and find root of interpolating polynomial
            # Using Lagrange form for inverse quadratic:
            # x = xa*ya*(yb-yc)/((ya-yb)*(ya-yc)) + xb*yb*(ya-yc)/((yb-ya)*(yb-yc)) + xc*yc*(ya-yb)/((yc-ya)*(yc-yb))
            
            if ya == yb or ya == yc or yb == yc:
                # Degenerate, use bisection
                xc_new = (xa + xb) / 2.0
            else:
                # Inverse quadratic interpolation
                term1 = xa * ya * (yb - yc) / ((ya - yb) * (ya - yc))
                term2 = xb * yb * (ya - yc) / ((yb - ya) * (yb - yc))
                term3 = xc * yc * (ya - yb) / ((yc - ya) * (yc - yb))
                xc_new = term1 + term2 + term3
                
                # Ensure xc_new is in bracket
                if not (xa < xc_new < xb):
                    xc_new = (xa + xb) / 2.0
            
            yc_new = eval_f(xc_new)
            if yc_new is None:
                break
            if abs(yc_new) <= 1e-8:
                return xc_new
            
            # Update points for next IQUAD
            # Keep the 3 most "useful" points
            # Simple strategy: replace the point with the largest |y|
            points = [(xa, ya), (xb, yb), (xc, yc)]
            new_point = (xc_new, yc_new)
            
            # Find which point to replace: the one with largest |y|
            max_idx = max(range(3), key=lambda i: abs(points[i][1]))
            points[max_idx] = new_point
            
            # Sort by x for consistency
            points.sort(key=lambda p: p[0])
            xa, ya = points[0]
            xb, yb = points[1]
            xc, yc = points[2]
            
            xc = xc_new
            yc = yc_new
        
        # After loop, return best estimate
        # Use the midpoint of the final bracket if available
        if b is not None and a is not None:
            return (xa + xb) / 2.0
        return xc
    
    # If no bracket found, try Newton's method from x0
    x = float(x0)
    fx = best_f
    best_abs = abs(fx)
    best_x = x
    
    for _ in range(40):
        if _time_left() < 1.0:
            break
        if best_abs <= 1e-8:
            return best_x
        
        h = 1e-7 * max(1.0, abs(x))
        fx_plus = eval_f(x + h)
        if fx_plus is None:
            break
        
        d = (fx_plus - fx) / h
        if d == 0 or not _is_finite(d):
            break
        
        x_new = x - fx / d
        if not _is_finite(x_new):
            break
        
        fx_new = eval_f(x_new)
        if fx_new is None:
            break
        
        if abs(fx_new) < best_abs:
            best_x = x_new
            best_abs = abs(fx_new)
        
        # Damping if step is too large
        if abs(x_new - x) > 10 * max(1.0, abs(x)):
            x_new = x + 0.5 * (x_new - x)
            fx_new = eval_f(x_new)
            if fx_new is None:
                break
        
        x = x_new
        fx = fx_new
    
    # Final polish with tighter tolerance
    if best_abs > 1e-8:
        for _ in range(20):
            if _time_left() < 1.0:
                break
            if abs(fx) <= 1e-12:
                break
            h = 1e-8 * max(1.0, abs(x))
            fx_plus = eval_f(x + h)
            if fx_plus is None:
                break
            d = (fx_plus - fx) / h
            if d == 0 or not _is_finite(d):
                break
            x = x - fx / d
            if not _is_finite(x):
                break
            fx = eval_f(x)
            if fx is None:
                break
    
    return best_x
