"""Root finder using a hybrid secant-Newton with adaptive bracketing and deadline guard."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 for a smooth polynomial f.
    Uses a combination of secant method, Newton refinement, and bracketing.
    """
    deadline = time.time() + 110.0  # safety margin before 120s limit
    x = float(x0)
    
    # First, check if x0 is already a root
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x
    
    # Try secant method with two initial points
    h = 1e-6 * max(1.0, abs(x))
    x1 = x + h
    f1 = f(x1)
    
    if abs(f1) <= 1e-8:
        return x1
    
    # Track best result so far
    best_x = x if abs(fx) < abs(f1) else x1
    best_f = min(abs(fx), abs(f1))
    
    # Secant method iterations
    x_prev = x
    f_prev = fx
    x_curr = x1
    f_curr = f1
    
    for i in range(40):
        if time.time() > deadline:
            break
            
        # Check convergence
        if abs(f_curr) <= 1e-8:
            return x_curr
        
        # Update best
        if abs(f_curr) < best_f:
            best_f = abs(f_curr)
            best_x = x_curr
        
        # Secant step
        if f_curr == f_prev:
            # Degenerate case, try a different perturbation
            x_new = x_curr + 1e-4 * max(1.0, abs(x_curr))
            f_new = f(x_new)
            if abs(f_new) <= 1e-8:
                return x_new
            x_prev, f_prev = x_curr, f_curr
            x_curr, f_curr = x_new, f_new
            continue
            
        denom = f_curr - f_prev
        if abs(denom) < 1e-15:
            x_new = x_curr + 1e-4 * max(1.0, abs(x_curr))
            f_new = f(x_new)
            if abs(f_new) <= 1e-8:
                return x_new
            x_prev, f_prev = x_curr, f_curr
            x_curr, f_curr = x_new, f_new
            continue
        
        x_new = x_curr - f_curr * (x_curr - x_prev) / denom
        
        # Check for non-finite
        if not math.isfinite(x_new):
            x_new = x_curr + 1e-4 * max(1.0, abs(x_curr))
            f_new = f(x_new)
            if abs(f_new) <= 1e-8:
                return x_new
            x_prev, f_prev = x_curr, f_curr
            x_curr, f_curr = x_new, f_new
            continue
        
        f_new = f(x_new)
        
        if not math.isfinite(f_new):
            x_new = x_curr + 1e-4 * max(1.0, abs(x_curr))
            f_new = f(x_new)
            if abs(f_new) <= 1e-8:
                return x_new
            x_prev, f_prev = x_curr, f_curr
            x_curr, f_curr = x_new, f_new
            continue
        
        if abs(f_new) <= 1e-8:
            return x_new
        
        # Update best
        if abs(f_new) < best_f:
            best_f = abs(f_new)
            best_x = x_new
        
        # Check for sign change - if so, we have a bracket
        if f_curr * f_new < 0:
            # We have a bracket [x_curr, x_new]
            a, b = x_curr, x_new
            fa, fb = f_curr, f_new
            
            # Bisection with secant acceleration
            for j in range(30):
                if time.time() > deadline:
                    break
                
                # Try secant within bracket
                if fb != fa:
                    x_sec = a - fa * (a - b) / (fb - fa)
                    if x_sec > a and x_sec < b:
                        f_sec = f(x_sec)
                        if abs(f_sec) <= 1e-8:
                            return x_sec
                        if f_sec * fa < 0:
                            b, fb = x_sec, f_sec
                        else:
                            a, fa = x_sec, f_sec
                        continue
                
                # Bisection
                x_mid = 0.5 * (a + b)
                f_mid = f(x_mid)
                if abs(f_mid) <= 1e-8:
                    return x_mid
                if f_mid * fa < 0:
                    b, fb = x_mid, f_mid
                else:
                    a, fa = x_mid, f_mid
                
                if abs(b - a) < 1e-12 * max(1.0, abs(a), abs(b)):
                    # Return the midpoint or the one with smaller |f|
                    if abs(fa) <= abs(fb):
                        return a
                    else:
                        return b
            
            # Return best of bracket endpoints
            if abs(fa) <= abs(fb):
                return a
            else:
                return b
        
        x_prev, f_prev = x_curr, f_curr
        x_curr, f_curr = x_new, f_new
    
    # If secant didn't converge, try Newton's method from best_x
    x = best_x
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x
    
    for i in range(30):
        if time.time() > deadline:
            break
        
        h = 1e-7 * max(1.0, abs(x))
        d = (f(x + h) - fx) / h
        if abs(d) < 1e-15:
            break
        
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
        
        f_new = f(x_new)
        if not math.isfinite(f_new):
            break
        
        if abs(f_new) <= 1e-8:
            return x_new
        
        if abs(f_new) < abs(fx):
            x, fx = x_new, f_new
        else:
            # Damped Newton
            x_new = x - 0.5 * fx / d
            f_new = f(x_new)
            if not math.isfinite(f_new):
                break
            if abs(f_new) <= 1e-8:
                return x_new
            x, fx = x_new, f_new
    
    # Final fallback: return best_x
    return best_x
# EVOLVE-BLOCK-END
