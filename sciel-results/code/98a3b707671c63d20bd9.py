"""Robust scalar root finder with deadline guard and multiple strategies."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, returning x with |f(x)| <= 1e-8.
    Uses a combination of Newton's method, bisection, and global search.
    Respects a wall-clock deadline.
    """
    DEADLINE = 145.0  # Safety margin before 150s limit
    START_TIME = time.time()
    
    def elapsed():
        return time.time() - START_TIME
    
    def is_finite(x):
        return math.isfinite(x)
    
    def safe_eval(x):
        """Safely evaluate f(x), returning None if it fails."""
        try:
            val = f(x)
            if is_finite(val):
                return val
        except:
            pass
        return None
    
    def is_good(x):
        """Check if x is a good root."""
        fx = safe_eval(x)
        return fx is not None and abs(fx) <= 1e-8
    
    def check_deadline():
        return elapsed() < DEADLINE
    
    # Initial evaluation
    x = float(x0)
    fx = safe_eval(x)
    
    # If already a root, return immediately
    if fx is not None and abs(fx) <= 1e-8:
        return x
    
    # Track best solution found
    best_x = x
    best_fx = abs(fx) if fx is not None else float('inf')
    
    def update_best(x_candidate, fx_candidate):
        nonlocal best_x, best_fx
        if fx_candidate is not None and abs(fx_candidate) < best_fx:
            best_x = x_candidate
            best_fx = abs(fx_candidate)
    
    # Strategy 1: Newton's method with finite difference derivative
    # Use adaptive step size for derivative
    newton_x = x
    newton_fx = fx
    
    for i in range(40):
        if not check_deadline():
            break
        
        if newton_fx is None or not is_finite(newton_x):
            break
        
        if abs(newton_fx) <= 1e-8:
            return newton_x
        
        # Compute derivative using central difference
        h = 1e-7 * max(1.0, abs(newton_x))
        fx_plus = safe_eval(newton_x + h)
        fx_minus = safe_eval(newton_x - h)
        
        if fx_plus is not None and fx_minus is not None:
            d = (fx_plus - fx_minus) / (2 * h)
        elif fx_plus is not None:
            d = (fx_plus - newton_fx) / h
        elif fx_minus is not None:
            d = (newton_fx - fx_minus) / h
        else:
            break
        
        if d == 0 or not is_finite(d):
            break
        
        # Damped Newton step
        step = newton_fx / d
        if not is_finite(step):
            break
        
        # Limit step size to prevent divergence
        max_step = 10.0 * max(1.0, abs(newton_x))
        if abs(step) > max_step:
            step = max_step * math.copysign(1.0, step)
        
        new_x = newton_x - step
        new_fx = safe_eval(new_x)
        
        if new_fx is None:
            # Try smaller step
            new_x = newton_x - step * 0.5
            new_fx = safe_eval(new_x)
        
        if new_fx is None:
            break
        
        update_best(new_x, new_fx)
        
        # Accept step if it reduces |f|
        if abs(new_fx) < abs(newton_fx):
            newton_x = new_x
            newton_fx = new_fx
        else:
            # Try half step
            half_step = step * 0.5
            half_x = newton_x - half_step
            half_fx = safe_eval(half_x)
            if half_fx is not None and abs(half_fx) < abs(newton_fx):
                newton_x = half_x
                newton_fx = half_fx
            else:
                # Try secant method instead
                break
    
    # Check if Newton found a root
    if newton_fx is not None and abs(newton_fx) <= 1e-8:
        return newton_x
    
    # Strategy 2: Bisection with bracketing search
    # First, try to find a bracket around x0
    bracket_found = False
    a, b = None, None
    fa, fb = None, None
    
    # Search for bracket by expanding interval
    delta = max(1.0, abs(x))
    for scale in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
        if not check_deadline():
            break
        
        left = x - scale * delta
        right = x + scale * delta
        
        fl = safe_eval(left)
        fr = safe_eval(right)
        
        if fl is not None and fr is not None:
            if fl == 0:
                return left
            if fr == 0:
                return right
            if fl * fr < 0:
                a, b = left, right
                fa, fb = fl, fr
                bracket_found = True
                break
    
    if not bracket_found:
        # Try asymmetric brackets
        for scale in [0.5, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
            if not check_deadline():
                break
            
            # Left-heavy
            left = x - scale * delta
            right = x + 0.1 * delta
            fl = safe_eval(left)
            fr = safe_eval(right)
            if fl is not None and fr is not None and fl * fr < 0:
                a, b = left, right
                fa, fb = fl, fr
                bracket_found = True
                break
            
            # Right-heavy
            left = x - 0.1 * delta
            right = x + scale * delta
            fl = safe_eval(left)
            fr = safe_eval(right)
            if fl is not None and fr is not None and fl * fr < 0:
                a, b = left, right
                fa, fb = fl, fr
                bracket_found = True
                break
    
    # Run bisection if bracket found
    if bracket_found:
        for i in range(80):
            if not check_deadline():
                break
            
            mid = (a + b) / 2
            fm = safe_eval(mid)
            
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
            if abs(b - a) < 1e-12:
                break
        
        # Return the better endpoint
        fa = safe_eval(a)
        fb = safe_eval(b)
        if fa is not None and fb is not None:
            if abs(fa) < abs(fb):
                return a
            else:
                return b
        elif fa is not None:
            return a
        elif fb is not None:
            return b
    
    # Strategy 3: Global search with random sampling
    # Sample points around x0 and in a wider range
    base_delta = max(1.0, abs(x))
    
    # Try a grid of points
    for scale in [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]:
        if not check_deadline():
            break
        
        for offset in [-1, -0.5, -0.2, 0, 0.2, 0.5, 1]:
            if not check_deadline():
                break
            
            candidate = x + offset * scale * base_delta
            fc = safe_eval(candidate)
            
            if fc is not None:
                update_best(candidate, fc)
                if abs(fc) <= 1e-8:
                    return candidate
    
    # Strategy 4: Secant method from best point
    if best_fx < float('inf') and is_finite(best_x):
        sec_x = best_x
        sec_fx = safe_eval(sec_x)
        
        if sec_fx is not None:
            # Find a second point
            h = 1e-4 * max(1.0, abs(sec_x))
            prev_x = sec_x - h
            prev_fx = safe_eval(prev_x)
            
            if prev_fx is not None:
                for i in range(30):
                    if not check_deadline():
                        break
                    
                    if sec_fx is None or prev_fx is None:
                        break
                    
                    denom = sec_fx - prev_fx
                    if denom == 0 or not is_finite(denom):
                        break
                    
                    sec_step = sec_x * sec_fx - prev_x * prev_fx
                    sec_step /= denom
                    
                    if not is_finite(sec_step):
                        break
                    
                    # Limit step
                    max_step = 10.0 * max(1.0, abs(sec_x))
                    if abs(sec_step) > max_step:
                        sec_step = max_step * math.copysign(1.0, sec_step)
                    
                    new_x = sec_x - sec_step
                    new_fx = safe_eval(new_x)
                    
                    if new_fx is None:
                        break
                    
                    update_best(new_x, new_fx)
                    
                    if abs(new_fx) <= 1e-8:
                        return new_x
                    
                    prev_x, prev_fx = sec_x, sec_fx
                    sec_x, sec_fx = new_x, new_fx
    
    # Final check: return best found
    if is_finite(best_x) and best_fx <= 1e-8:
        return best_x
    
    # If best_fx is close, try to polish
    if is_finite(best_x) and best_fx < 1e-4:
        # Try a few Newton steps from best_x
        px = best_x
        pfx = safe_eval(px)
        for i in range(10):
            if not check_deadline():
                break
            if pfx is None:
                break
            if abs(pfx) <= 1e-8:
                return px
            h = 1e-7 * max(1.0, abs(px))
            fp = safe_eval(px + h)
            fm = safe_eval(px - h)
            if fp is not None and fm is not None:
                d = (fp - fm) / (2 * h)
            elif fp is not None:
                d = (fp - pfx) / h
            elif fm is not None:
                d = (pfx - fm) / h
            else:
                break
            if d == 0 or not is_finite(d):
                break
            step = pfx / d
            if abs(step) > 10.0 * max(1.0, abs(px)):
                step = 10.0 * max(1.0, abs(px)) * math.copysign(1.0, step)
            px = px - step
            pfx = safe_eval(px)
            if pfx is None:
                break
            if abs(pfx) <= 1e-8:
                return px
    
    # Return best found, or original x0 if nothing better
    if is_finite(best_x):
        return best_x
    return x
# EVOLVE-BLOCK-END
