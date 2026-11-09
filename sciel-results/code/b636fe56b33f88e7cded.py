import math
import time

def find_root(f, x0):
    tol = 1e-8
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    x = float(x0)
    fx = f(x)
    if abs(fx) <= tol:
        return x
    
    # Try to find a bracket [a, b] where f(a) * f(b) < 0
    # Start with small steps, then expand geometrically
    a = x
    fa = fx
    
    # First check nearby points for a sign change
    deltas = [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    
    bracket_found = False
    b = None
    fb = None
    
    for delta in deltas:
        if time.time() > deadline:
            return x  # Return best so far
        
        # Check x + delta
        x_plus = x + delta
        f_plus = f(x_plus)
        if abs(f_plus) <= tol:
            return x_plus
        if fa * f_plus < 0:
            b = x_plus
            fb = f_plus
            bracket_found = True
            break
        
        # Check x - delta
        x_minus = x - delta
        f_minus = f(x_minus)
        if abs(f_minus) <= tol:
            return x_minus
        if fa * f_minus < 0:
            b = x_minus
            fb = f_minus
            bracket_found = True
            break
    
    # If no bracket found with initial deltas, try larger geometric expansion
    if not bracket_found:
        step = 1.0
        for i in range(50):
            if time.time() > deadline:
                return x
            
            # Try expanding in both directions
            for sign in [1, -1]:
                x_try = x + sign * step
                f_try = f(x_try)
                if abs(f_try) <= tol:
                    return x_try
                if fa * f_try < 0:
                    b = x_try
                    fb = f_try
                    bracket_found = True
                    break
            if bracket_found:
                break
            step *= 2.0
    
    if not bracket_found:
        # No bracket found; return the point with smallest |f|
        return x
    
    # Hybrid Secant/Bisection on [a, b]
    lo = a
    hi = b
    flo = fa
    fhi = fb
    
    # Ensure a < b for clarity (not required for bisection logic)
    if lo > hi:
        lo, hi = hi, lo
        flo, fhi = fhi, flo

    # Hybrid Solver: Try Secant, fallback to Bisection
    for i in range(100):
        if time.time() > deadline:
            return (lo + hi) / 2.0
            
        if abs(hi - lo) < 1e-12:
            break
            
        # Attempt Secant Step
        denom = fhi - flo
        if abs(denom) > 1e-300:
            x_secant = hi - fhi * (hi - lo) / denom
            # Check if x_secant is inside (lo, hi)
            if lo < x_secant < hi:
                x_new = x_secant
            else:
                x_new = (lo + hi) / 2.0
        else:
            x_new = (lo + hi) / 2.0
            
        f_new = f(x_new)
        
        if abs(f_new) <= tol:
            return x_new
            
        # Update Bracket
        if flo * f_new < 0:
            hi = x_new
            fhi = f_new
        else:
            lo = x_new
            flo = f_new
            
    return (lo + hi) / 2.0
