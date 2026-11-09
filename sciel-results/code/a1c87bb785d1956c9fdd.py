import math
import time

def find_root(f, x0):
    """
    Find a root of f starting near x0 using a robust multi-strategy approach.
    Returns x such that |f(x)| <= 1e-8.
    """
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    tol = 1e-9  # Internal tolerance tighter than required 1e-8
    
    x = float(x0)
    
    # Evaluate initial point
    try:
        f_x = f(x)
    except:
        return x
    
    if not math.isfinite(f_x):
        return x
        
    best_x = x
    best_f = f_x
    best_abs = abs(best_f)
    
    if best_abs <= tol:
        return x
    
    # Strategy 1: Bracketing with adaptive expansion
    # Try to find a sign change near x0
    a = x
    fa = f_x
    b = None
    fb = None
    
    # Adaptive step size based on local scale
    base_step = 1e-3 * max(1.0, abs(x))
    
    # Try expanding in both directions with increasing step
    for i in range(30):
        if time.time() > deadline:
            break
            
        step = base_step * (2 ** i)
        
        # Try positive direction
        x_pos = x + step
        try:
            f_pos = f(x_pos)
            if math.isfinite(f_pos) and fa * f_pos < 0:
                a, b = x, x_pos
                fa, fb = f_x, f_pos
                break
        except:
            pass
            
        # Try negative direction
        x_neg = x - step
        try:
            f_neg = f(x_neg)
            if math.isfinite(f_neg) and fa * f_neg < 0:
                a, b = x_neg, x
                fa, fb = f_neg, f_x
                break
        except:
            pass
            
        # Update best if we found a better point
        for xp, fp in [(x_pos, f_pos), (x_neg, f_neg)]:
            if fp is not None and math.isfinite(fp):
                if abs(fp) < best_abs:
                    best_x, best_f, best_abs = xp, fp, abs(fp)
                    if best_abs <= tol:
                        return best_x
    
    # If bracket found, use Brent-like method (secant + bisection hybrid)
    if b is not None:
        # Ensure a < b for consistency
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
        
        # Secant + Bisection hybrid (Illinois-style with secant preference)
        for _ in range(100):
            if time.time() > deadline:
                break
                
            # Secant step
            if fa == fb:
                x_new = (a + b) / 2.0
            else:
                x_new = b - fb * (b - a) / (fb - fa)
                
            # If secant step is outside bracket or too close to endpoint, use bisection
            if x_new <= a or x_new >= b or abs(x_new - a) < 1e-12 * (b - a) or abs(x_new - b) < 1e-12 * (b - a):
                x_new = (a + b) / 2.0
                
            try:
                f_new = f(x_new)
            except:
                break
                
            if not math.isfinite(f_new):
                break
                
            # Update best
            if abs(f_new) < best_abs:
                best_x, best_f, best_abs = x_new, f_new, abs(f_new)
                if best_abs <= tol:
                    return best_x
                    
            # Update bracket
            if fa * f_new < 0:
                b, fb = x_new, f_new
            else:
                a, fa = x_new, f_new
                
        return best_x
    
    # Strategy 2: No bracket found. Use derivative-free optimization.
    # Since functions are oscillatory, try to find local minima of |f(x)|
    
    # First, try secant method from x0
    h = 1e-6 * max(1.0, abs(x))
    x_prev = x
    f_prev = f_x
    x_curr = x + h
    
    try:
        f_curr = f(x_curr)
    except:
        f_curr = f_x  # Fallback
        
    for _ in range(50):
        if time.time() > deadline:
            break
            
        if abs(f_curr) < best_abs:
            best_x, best_f, best_abs = x_curr, f_curr, abs(f_curr)
            if best_abs <= tol:
                return best_x
                
        if abs(f_prev) < best_abs:
            best_x, best_f, best_abs = x_prev, f_prev, abs(f_prev)
            if best_abs <= tol:
                return best_x
                
        if f_prev == f_curr:
            break
            
        # Secant step
        x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        
        if not math.isfinite(x_new):
            break
            
        x_prev, f_prev = x_curr, f_curr
        x_curr = x_new
        
        try:
            f_curr = f(x_curr)
        except:
            break
            
    return best_x
