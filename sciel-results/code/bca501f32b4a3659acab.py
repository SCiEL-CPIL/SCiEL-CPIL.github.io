import numpy as np
from scipy.optimize import brentq
import time

def find_root(f, x0):
    x0 = float(x0)
    start_time = time.time()
    
    # Helper to check time limit
    def time_exceeded():
        return (time.time() - start_time) > 110.0 # 10s margin
    
    # 1. Check initial point
    try:
        f0 = f(x0)
    except Exception:
        return x0 # Return initial if f fails, though task implies f is valid
    
    if abs(f0) <= 1e-8:
        return x0
    
    # 2. Bracketing Phase
    # We search for an interval [a, b] such that f(a)*f(b) < 0
    # We expand the search radius geometrically from x0
    
    # Initial step size
    # Scale with x0 to handle different magnitudes
    h = 1e-2 * max(1.0, abs(x0))
    
    fa = None
    fb = None
    a = None
    b = None
    
    # Keep track of the best point found (min |f(x)|) in case bracketing fails
    best_x = x0
    best_f = f0
    
    # We will try to find a bracket by checking x0 +/- h
    # If we find a sign change, we stop.
    # We expand h by factor of 2 each time.
    # Max 30 expansions to cover huge ranges or very small periods.
    
    for i in range(30):
        if time_exceeded():
            break
            
        x_plus = x0 + h
        x_minus = x0 - h
        
        try:
            fp = f(x_plus)
            fm = f(x_minus)
        except Exception:
            # If f fails, shrink or break? Usually f is well-behaved.
            # If f fails, we might be out of domain. Let's try shrinking?
            # But for oscillatory functions, domain is likely all reals or large.
            # Let's just break and return best.
            break
        
        # Update best point
        if abs(fp) < abs(best_f):
            best_f = fp
            best_x = x_plus
        if abs(fm) < abs(best_f):
            best_f = fm
            best_x = x_minus
        
        # Check for sign change between x0 and x_plus
        if f0 * fp < 0:
            a = x0
            b = x_plus
            fa = f0
            fb = fp
            break
        
        # Check for sign change between x_minus and x0
        if fm * f0 < 0:
            a = x_minus
            b = x0
            fa = fm
            fb = f0
            break
            
        # Check for sign change between x_minus and x_plus (if we skipped x0?)
        # Actually, if f0 and fp have same sign, and fm and f0 have same sign,
        # it's possible fm and fp have different signs if the function oscillates
        # and x0 is in a "flat" region or we jumped over a zero.
        # But since we are checking adjacent points in a sequence of expansions,
        # checking x0 to neighbors is the primary way.
        # However, if the period is very small, we might miss it.
        # The geometric expansion ensures we eventually cover the period.
        
        # Expand step size
        h *= 2.0
        
        # Safety: if h becomes too large, we might hit overflow or domain issues
        if h > 1e15:
            break
            
    # 3. Convergence Phase
    if a is not None and b is not None:
        # We have a bracket [a, b]
        try:
            # Use brentq for robust, fast convergence
            # xtol=1e-14 is very tight, ensures |f(x)| is small
            root = brentq(f, a, b, args=(), xtol=1e-14, rtol=1e-14, maxiter=100)
            # Verify the result
            if abs(f(root)) <= 1e-8:
                return root
            else:
                # Fallback if brentq returned something weird
                return best_x
        except Exception:
            # If brentq fails, try bisection manually or return best
            # Bisection is robust
            try:
                left, right = a, b
                for _ in range(100):
                    mid = (left + right) / 2.0
                    fm = f(mid)
                    if abs(fm) <= 1e-8:
                        return mid
                    if fa * fm < 0:
                        right = mid
                        fb = fm
                    else:
                        left = mid
                        fa = fm
                return (left + right) / 2.0
            except Exception:
                return best_x
    else:
        # No bracket found. Return the best point found.
        # This is a failure case for strict root finding, but better than crashing.
        return best_x
