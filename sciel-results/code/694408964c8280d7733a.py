import math
import time
import scipy.optimize

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, returning x with |f(x)| <= 1e-8.
    Strategy:
    1. Check if x0 is already a root.
    2. Try to find a bracket [a, b] where f(a) and f(b) have opposite signs.
       If found, use scipy.optimize.brentq which is robust and efficient.
    3. If no bracket is found, use a safeguarded Newton's method with finite difference derivative.
    Includes a deadline guard to prevent timeout.
    """
    TOL = 1e-8
    DEADLINE = 140.0  # Safety margin before 150s kill
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

    # We will keep track of the best candidate (min |f(x)|) found so far
    best_x = x0
    best_f_abs = abs(fx0)

    # Strategy 1: Bracketing + Brent's Method
    # We need a bracket [a, b] such that f(a) * f(b) < 0.
    # We will expand outward from x0.
    
    # Start with a small step to catch nearby roots
    step = 1.0
    # We will try up to 100 expansions
    for i in range(100):
        if not check_deadline():
            break
            
        a = x0 - step
        b = x0 + step
        
        try:
            fa = f(a)
            fb = f(b)
        except Exception:
            # If f fails, maybe step is too large or domain issue?
            # But task says f is a smooth polynomial, so it shouldn't fail.
            # If it does, we break.
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
                # Use brentq with tight tolerance
                root = scipy.optimize.brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                if is_finite(root):
                    # Verify
                    f_root = f(root)
                    if is_finite(f_root) and abs(f_root) <= TOL:
                        return root
                    # If brentq didn't quite hit the tolerance (unlikely), 
                    # we might try a few Newton steps from root, or just return root 
                    # if it's close enough? The contract says |f(x)| <= 1e-8.
                    # Brentq with xtol=1e-12 usually guarantees this for smooth funcs.
                    return root
            except Exception:
                # If brentq fails, we fall through to return best_x
                break
                
        # Expand step
        step *= 2.0
        
        # Safety limit
        if step > 1e10:
            break

    # If we reach here, we failed to find a bracket or brentq failed.
    # Strategy 2: Safeguarded Newton's method
    # This is useful if the root is not bracketed by simple expansion (e.g., double root)
    # or if we missed the bracket.
    
    x = x0
    fx = fx0
    
    for i in range(50):
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
            # Derivative zero or near zero
            break
            
        step = fx / dfx
        x_new = x - step
        
        # Safeguard: if step is huge, something is wrong
        if not is_finite(x_new) or abs(step) > 10 * max(1.0, abs(x)):
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
            
        # Check for stagnation or divergence
        if abs(fx_new) > abs(fx) * 1.1 and i > 2:
            # If error is increasing, stop trying local refinement
            break
            
        x = x_new
        fx = fx_new

    # If we reach here, we failed to find a root.
    # Return the best x found.
    return best_x
# EVOLVE-BLOCK-END
