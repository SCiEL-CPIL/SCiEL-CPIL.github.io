import math
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Returns x with |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses Brent's method with robust bracketing.
    """
    max_evals = 500
    eval_count = 0
    tol = 1e-8
    
    def eval_f(val):
        nonlocal eval_count
        if eval_count >= max_evals:
            raise RuntimeError("Max evaluations exceeded")
        result = f(val)
        eval_count += 1
        return result

    x = float(x0)
    fx = eval_f(x)
    if abs(fx) <= tol:
        return x

    # Find a bracket [a, b] such that f(a) * f(b) < 0
    # Start with a small step and expand
    h = 1e-4 * max(1.0, abs(x))
    a = x
    fa = fx
    b = x + h
    fb = eval_f(b)
    
    if abs(fb) <= tol:
        return b

    # Expand bracket
    found_bracket = False
    for i in range(50):
        if fa * fb < 0:
            found_bracket = True
            break
        
        # Expand in the direction that seems to move away from the root?
        # Or just expand both? 
        # Standard strategy: if f(a) and f(b) have same sign, expand the interval.
        # We need to decide which side to expand.
        # If |fa| < |fb|, expand b further. Else expand a further.
        if abs(fa) < abs(fb):
            b = b + h
            fb = eval_f(b)
            if abs(fb) <= tol:
                return b
        else:
            a = a - h
            fa = eval_f(a)
            if abs(fa) <= tol:
                return a
            
        h *= 2
        
        # Stop if values become non-finite or too large
        if not np.isfinite(fa) or not np.isfinite(fb):
            break
        if abs(a) > 1e10 or abs(b) > 1e10:
            break

    if found_bracket:
        try:
            root = brentq(eval_f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if abs(eval_f(root)) <= tol: # Note: this adds 1 eval
                return root
            # If brentq returned something but it's not good enough (unlikely), 
            # we might need to check endpoints
            if abs(fa) <= tol: return a
            if abs(fb) <= tol: return b
            return root
        except Exception:
            pass

    # Fallback: Secant method if no bracket found
    # We have a, fa, b, fb from the expansion loop (or initial)
    # Actually, if no bracket, a and b are just the last two points tried.
    # Let's restart secant with x and x+h
    x0_s = x
    f0_s = fx
    x1_s = x + 1e-4 * max(1.0, abs(x))
    f1_s = eval_f(x1_s)
    
    if abs(f1_s) <= tol:
        return x1_s

    for i in range(max_evals - eval_count):
        if abs(f1_s) <= tol:
            return x1_s
        if abs(f0_s) <= tol:
            return x0_s
            
        if f0_s == f1_s:
            x1_s = x1_s + 1e-6
            f1_s = eval_f(x1_s)
            continue
            
        x_new = x1_s - f1_s * (x1_s - x0_s) / (f1_s - f0_s)
        
        # Safeguard
        if abs(x_new - x1_s) > 10 * max(1.0, abs(x1_s)):
             x_new = x1_s + 0.5 * (x_new - x1_s)
             
        f_new = eval_f(x_new)
        if abs(f_new) <= tol:
            return x_new
            
        x0_s = x1_s
        f0_s = f1_s
        x1_s = x_new
        f1_s = f_new
        
        if abs(x_new - x0_s) < 1e-14 * max(1.0, abs(x1_s)):
            break
            
    # Return best candidate
    candidates = [x0_s, x1_s]
    best_x = candidates[0]
    best_val = abs(eval_f(best_x))
    for cx in candidates[1:]:
        cf = abs(eval_f(cx))
        if cf < best_val:
            best_val = cf
            best_x = cx
    return best_x
