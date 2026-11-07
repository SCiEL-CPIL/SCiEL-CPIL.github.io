import math
import time
import scipy.optimize

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, returning x with |f(x)| <= 1e-8.
    Strategy:
    1. Check if x0 is already a root.
    2. Attempt Newton-Raphson iterations.
       - Use central difference for derivative.
       - Accept step only if it reduces |f(x)|.
       - If a bracket is formed (sign change), switch to Brent's method.
       - If Newton stalls or diverges, switch to a bracketing search.
    3. Fallback to bracketing search + Brent's method if Newton fails.
    Includes a deadline guard to prevent timeout.
    """
    TOL = 1e-8
    DEADLINE = 115.0  # Safety margin before 150s kill
    START_TIME = time.time()
    
    def is_finite(x):
        return math.isfinite(x)

    def check_deadline():
        return (time.time() - START_TIME) < DEADLINE

    # Initial check
    try:
        fx0 = f(x0)
    except Exception:
        return x0
        
    if not is_finite(fx0):
        return x0

    if abs(fx0) <= TOL:
        return x0

    best_x = x0
    best_f_abs = abs(fx0)

    # Strategy 1: Newton-Raphson with safeguards
    # We track a potential bracket [a, b] if signs differ
    a = x0
    fa = fx0
    b = None
    fb = None
    
    x = x0
    fx = fx0
    
    # Max Newton iterations to try before falling back
    max_newton_iters = 20
    
    for i in range(max_newton_iters):
        if not check_deadline():
            break
            
        if abs(fx) <= TOL:
            return x
            
        # Estimate derivative using central differences
        h = 1e-5 * max(1.0, abs(x))
        
        try:
            fx_h = f(x + h)
            fx_mh = f(x - h)
        except Exception:
            break
            
        if not (is_finite(fx_h) and is_finite(fx_mh)):
            break
            
        dfx = (fx_h - fx_mh) / (2 * h)
        
        if abs(dfx) < 1e-15:
            # Derivative zero, Newton fails
            break
            
        # Newton step
        step = fx / dfx
        x_new = x - step
        
        if not is_finite(x_new):
            break
            
        try:
            fx_new = f(x_new)
        except Exception:
            break
            
        if not is_finite(fx_new):
            break
            
        # Update best
        if abs(fx_new) < best_f_abs:
            best_f_abs = abs(fx_new)
            best_x = x_new
            
        # Check for convergence
        if abs(fx_new) <= TOL:
            return x_new
            
        # Check for bracket formation
        # If f(x) and f(x_new) have opposite signs, we have a bracket
        if fa * fx_new < 0:
            # Bracket is [a, x_new]
            a_bracket = a
            b_bracket = x_new
            try:
                root = scipy.optimize.brentq(f, a_bracket, b_bracket, xtol=1e-12, rtol=1e-12, maxiter=50)
                if is_finite(root):
                    f_root = f(root)
                    if is_finite(f_root) and abs(f_root) <= TOL:
                        return root
                    return root
            except Exception:
                pass
        
        # If f(x_new) and f(x) have opposite signs (and we didn't catch it above because 'a' was updated?)
        # Actually, if fx * fx_new < 0, we have a bracket [x, x_new]
        if fx * fx_new < 0:
            a_bracket = x
            b_bracket = x_new
            try:
                root = scipy.optimize.brentq(f, a_bracket, b_bracket, xtol=1e-12, rtol=1e-12, maxiter=50)
                if is_finite(root):
                    f_root = f(root)
                    if is_finite(f_root) and abs(f_root) <= TOL:
                        return root
                    return root
            except Exception:
                pass

        # Safeguard: Only accept the Newton step if it reduces the error
        # This prevents divergence. If it doesn't reduce error, we break and try bracketing.
        if abs(fx_new) < abs(fx):
            x = x_new
            fx = fx_new
            # Update the 'a' for potential future bracketing? 
            # If we keep moving in one direction and sign doesn't change, we are far.
            # We keep 'a' as the starting point x0 to see if we cross it? 
            # Actually, for bracketing, we need two points with opposite signs.
            # If we just keep accepting Newton steps, we might fly off to infinity.
            # If abs(fx_new) < abs(fx), we are getting closer to the root in value, 
            # but not necessarily in position if the function is flat? 
            # For polynomials, if |f| decreases, we are generally moving towards a root.
            
            # Update a to be the previous x if signs are same? 
            # If signs are same, we haven't bracketed.
            # We don't update 'a' here to keep the original x0 as a reference for bracketing 
            # against the current x? 
            # If f(x0) and f(x_current) have same sign, no bracket.
            # If they have opposite sign, we would have caught it in the checks above.
            
        else:
            # Newton step did not improve.
            # Try a damped step? Or break and bracket.
            # Given the risk, break and use robust bracketing.
            break

    # Strategy 2: Robust Bracketing Search
    # If Newton failed or wasn't used, we need to find a bracket.
    # We search around x0 (or best_x if it's better).
    # Start with the best_x found so far.
    center = best_x
    step = 1.0
    
    # Try to find a bracket [c - step, c + step]
    for i in range(100):
        if not check_deadline():
            break
            
        a = center - step
        b = center + step
        
        try:
            fa = f(a)
            fb = f(b)
        except Exception:
            # If f fails, expand? Or break.
            break
            
        if not (is_finite(fa) and is_finite(fb)):
            break
            
        # Update best
        if abs(fa) < best_f_abs:
            best_f_abs = abs(fa)
            best_x = a
        if abs(fb) < best_f_abs:
            best_f_abs = abs(fb)
            best_x = b
            
        if fa == 0:
            return a
        if fb == 0:
            return b
            
        if fa * fb < 0:
            # Bracket found!
            try:
                root = scipy.optimize.brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                if is_finite(root):
                    f_root = f(root)
                    if is_finite(f_root) and abs(f_root) <= TOL:
                        return root
                    return root
            except Exception:
                break
                
        # Expand step
        step *= 2.0
        
        # Safety limit
        if step > 1e10:
            break

    # If we reach here, we failed to find a root.
    # Return the best x found.
    return best_x
# EVOLVE-BLOCK-END
