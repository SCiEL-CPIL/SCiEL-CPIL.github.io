"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust scalar root finder for smooth polynomial equations.
    Uses a multi-strategy approach:
    1. Check initial point.
    2. Try Newton's method with adaptive step (damped) for fast local convergence.
    3. If Newton fails or diverges, try Secant method.
    4. If that fails, try Bisection on a bracket found by expanding from x0.
    5. Deadline guard to ensure we return within time limits.
    """
    start_time = time.time()
    DEADLINE = start_time + 100.0  # Leave 20s margin for the 120s limit
    TOL = 1e-12
    MAX_CALLS = 450  # Stay under 500 limit
    
    def now():
        return time.time()
    
    def is_valid(x):
        return math.isfinite(x)
    
    def check_and_return(x):
        """Check if x is a valid root and return it, or raise/return None."""
        if not is_valid(x):
            return None
        try:
            fx = f(x)
            if not math.isfinite(fx):
                return None
            if abs(fx) <= TOL:
                return x
        except:
            return None
        return None
    
    # Initial evaluation
    x = float(x0)
    try:
        fx = f(x)
    except:
        # If initial point fails, try nearby points
        for offset in [1e-4, -1e-4, 1e-3, -1e-3, 1e-2, -1e-2]:
            x_try = x0 + offset
            try:
                fx_try = f(x_try)
                if math.isfinite(fx_try):
                    x = x_try
                    fx = fx_try
                    break
            except:
                continue
        else:
            return x0  # Give up, return initial
    
    if not math.isfinite(fx):
        # Try to find a finite point
        for offset in [1e-4, -1e-4, 1e-3, -1e-3, 1e-2, -1e-2, 1.0, -1.0]:
            x_try = x0 + offset
            try:
                fx_try = f(x_try)
                if math.isfinite(fx_try):
                    x = x_try
                    fx = fx_try
                    break
            except:
                continue
        else:
            return x0
    
    # Check if we already have a root
    if abs(fx) <= TOL:
        return x
    
    # Strategy 1: Damped Newton's method
    # Use central difference for derivative initially, then forward if close
    h = 1e-8 * max(1.0, abs(x))
    
    best_x = x
    best_fx = abs(fx)
    
    for i in range(60):
        if now() > DEADLINE:
            # Return best found
            if best_fx <= 1e-8:
                return best_x
            # Try one more evaluation of best_x
            return best_x
        
        # Compute derivative
        try:
            if h > 1e-10:
                # Central difference
                f_plus = f(x + h)
                f_minus = f(x - h)
                if not (math.isfinite(f_plus) and math.isfinite(f_minus)):
                    h *= 0.5
                    continue
                df = (f_plus - f_minus) / (2 * h)
            else:
                # Forward difference
                f_plus = f(x + h)
                if not math.isfinite(f_plus):
                    h *= 0.5
                    continue
                df = (f_plus - fx) / h
        except:
            break
        
        if not math.isfinite(df) or abs(df) < 1e-15:
            # Derivative too small or invalid, try secant
            break
        
        # Newton step
        dx = fx / df
        x_new = x - dx
        
        # Damping: if step is too large, damp it
        if abs(dx) > abs(x) * 0.5 + 1.0:
            x_new = x - 0.5 * dx
        
        if not is_valid(x_new):
            break
        
        try:
            fx_new = f(x_new)
        except:
            break
        
        if not math.isfinite(fx_new):
            break
        
        # Update best
        if abs(fx_new) < best_fx:
            best_fx = abs(fx_new)
            best_x = x_new
        
        if abs(fx_new) <= TOL:
            return x_new
        
        # Check for convergence
        if abs(x_new - x) < 1e-15 * max(1.0, abs(x_new)):
            # Converged but not to TOL?
            if abs(fx_new) <= 1e-8:
                return x_new
            break
        
        x = x_new
        fx = fx_new
        
        # Adapt h
        h = 1e-8 * max(1.0, abs(x))
    
    if best_fx <= 1e-8:
        return best_x
    
    # Strategy 2: Secant method
    x1 = x
    f1 = fx
    x0_sec = x - 1e-4 * max(1.0, abs(x))
    try:
        f0_sec = f(x0_sec)
    except:
        f0_sec = None
    
    if f0_sec is not None and math.isfinite(f0_sec):
        for i in range(50):
            if now() > DEADLINE:
                if best_fx <= 1e-8:
                    return best_x
                return best_x
            
            if abs(f0_sec) == abs(f1):
                break
            
            dx_sec = (x1 - x0_sec) * f1 / (f1 - f0_sec)
            x_new = x1 - dx_sec
            
            if not is_valid(x_new):
                break
            
            try:
                fx_new = f(x_new)
            except:
                break
            
            if not math.isfinite(fx_new):
                break
            
            if abs(fx_new) < best_fx:
                best_fx = abs(fx_new)
                best_x = x_new
            
            if abs(fx_new) <= TOL:
                return x_new
            
            if abs(x_new - x1) < 1e-15 * max(1.0, abs(x_new)):
                if abs(fx_new) <= 1e-8:
                    return x_new
                break
            
            x0_sec, f0_sec = x1, f1
            x1, f1 = x_new, fx_new
    
    if best_fx <= 1e-8:
        return best_x
    
    # Strategy 3: Bisection with bracket finding
    # Try to find a bracket [a, b] where f(a) and f(b) have opposite signs
    a = x0
    fa = None
    try:
        fa = f(a)
    except:
        pass
    
    if fa is None or not math.isfinite(fa):
        # Try to find a valid starting point
        for offset in [1e-4, -1e-4, 1e-3, -1e-3, 1e-2, -1e-2, 0.1, -0.1, 1.0, -1.0]:
            a_try = x0 + offset
            try:
                fa_try = f(a_try)
                if math.isfinite(fa_try):
                    a = a_try
                    fa = fa_try
                    break
            except:
                continue
        else:
            return best_x if best_fx <= 1e-8 else x0
    
    # Expand to find a bracket
    b = None
    fb = None
    step = 1e-4
    max_expansions = 100
    
    for _ in range(max_expansions):
        if now() > DEADLINE:
            return best_x if best_fx <= 1e-8 else x0
        
        for sign in [1, -1]:
            b_try = a + sign * step
            try:
                fb_try = f(b_try)
            except:
                continue
            if not math.isfinite(fb_try):
                continue
            if fa * fb_try < 0:
                b = b_try
                fb = fb_try
                break
        if b is not None:
            break
        step *= 2
    
    if b is not None:
        # Bisection
        for i in range(100):
            if now() > DEADLINE:
                return best_x if best_fx <= 1e-8 else x0
            
            mid = (a + b) / 2.0
            try:
                fmid = f(mid)
            except:
                break
            
            if not math.isfinite(fmid):
                break
            
            if abs(fmid) < best_fx:
                best_fx = abs(fmid)
                best_x = mid
            
            if abs(fmid) <= TOL:
                return mid
            
            if fa * fmid <= 0:
                b = mid
                fb = fmid
            else:
                a = mid
                fa = fmid
            
            if abs(b - a) < 1e-15 * max(1.0, abs(a)):
                break
    
    # Final check
    if best_fx <= 1e-8:
        return best_x
    
    # Return the best candidate we have
    return best_x
# EVOLVE-BLOCK-END
