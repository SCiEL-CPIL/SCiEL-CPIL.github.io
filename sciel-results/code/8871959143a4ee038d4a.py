import numpy as np
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0
    
    x0 = float(x0)
    
    # 1. Single-shot polynomial interpolation
    # Choose step size h. 
    # If x0 is 0, h=1. If x0 is large, h should scale?
    # For local approximation, h should be small. 
    # Let's try h = 1e-3 * max(1, abs(x0))
    h = 1e-3 * max(1.0, abs(x0))
    
    # Points: x0 - 2h, x0 - h, x0, x0 + h, x0 + 2h
    xs = np.array([x0 - 2*h, x0 - h, x0, x0 + h, x0 + 2*h])
    try:
        ys = np.array([f(x) for x in xs])
    except Exception:
        # Fallback if f fails on these points
        return x0
        
    # Check for non-finite
    if not np.all(np.isfinite(ys)):
        return x0

    # Fit 4th degree polynomial
    # We can use polyfit or solve Vandermonde
    try:
        coeffs = np.polyfit(xs, ys, 4)
        roots = np.roots(coeffs)
    except Exception:
        # Fallback
        return x0

    # Filter real roots
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-10]
    
    best_x = x0
    best_f = abs(f(x0)) # We already have f(x0) in ys[2], so we can reuse? 
    # Actually, we called f(x0) in the list. Let's reuse it to save a call if we just check.
    # But we need to check the candidate.
    
    if real_roots:
        # Pick root closest to x0
        candidate = min(real_roots, key=lambda r: abs(r - x0))
        # Verify candidate
        f_cand = f(candidate)
        if np.isfinite(f_cand) and abs(f_cand) <= 1e-8:
            return candidate
        
        # If not good enough, try other real roots?
        # Or use candidate as start for Newton
        
        # If we haven't hit deadline, do a few Newton steps
        # Use the derivative from the polynomial or finite diff?
        # Finite diff is 2 calls per step. 
        # We have used 5 calls + 1 verification = 6 calls.
        # We can afford ~10 more calls (3 steps) easily.
        
        x = candidate
        fx = f_cand
        for _ in range(10):
            if time.time() > deadline - 5.0:
                break
            if abs(fx) <= 1e-12:
                return x
            h_new = 1e-6 * max(1.0, abs(x))
            d = (f(x + h_new) - fx) / h_new
            if d == 0:
                break
            x_new = x - fx / d
            if not np.isfinite(x_new):
                break
            x = x_new
            fx = f(x)
            if abs(fx) <= 1e-8:
                return x
        
        return x
        
    # If no real roots found or interpolation failed
    # Fall back to standard Newton from x0
    x = x0
    fx = f(x)
    for _ in range(50):
        if time.time() > deadline - 5.0:
            break
        if abs(fx) <= 1e-12:
            return x
        h_new = 1e-6 * max(1.0, abs(x))
        d = (f(x + h_new) - fx) / h_new
        if d == 0:
            break
        x = x - fx / d
        if not np.isfinite(x):
            break
        fx = f(x)
        if abs(fx) <= 1e-8:
            return x
            
    return x
