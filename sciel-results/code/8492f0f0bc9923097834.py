"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import brentq

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 145.0  # safety margin before 150s kill
    x0 = float(x0)
    
    def time_ok():
        return (time.time() - start_time) < TIME_LIMIT
    
    # Check initial point
    f0 = f(x0)
    if not np.isfinite(f0):
        f0 = float('inf')
    if abs(f0) <= 1e-8:
        return x0
    
    best_x = x0
    best_f = abs(f0)
    
    # Multi-scale sign-change scan using log-scan strategy
    # This minimizes bracketing calls by scanning at multiple scales
    # Scales chosen to cover a wide range efficiently
    scales = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    
    bracket = None
    
    for scale in scales:
        if not time_ok():
            break
        
        # Scan in both directions from x0
        for direction in [1, -1]:
            if not time_ok():
                break
            
            # Use log-spaced points for efficient scanning
            # Start close and expand outward
            n_points = 8
            for i in range(n_points):
                offset = direction * scale * (10 ** (i * 0.5))
                x_test = x0 + offset
                if not np.isfinite(x_test):
                    continue
                
                f_test = f(x_test)
                if not np.isfinite(f_test):
                    continue
                
                # Update best if this is closer to zero
                if abs(f_test) < best_f:
                    best_f = abs(f_test)
                    best_x = x_test
                
                if abs(f_test) <= 1e-8:
                    return x_test
                
                # Check for sign change with previous point
                if bracket is None:
                    # First point in this direction - store as reference
                    ref_x = x0
                    ref_f = f0
                else:
                    ref_x, ref_f = bracket[0], bracket[1]
                
                if ref_f * f_test < 0:
                    # Sign change found!
                    bracket = (min(x0, x_test), max(x0, x_test)) if direction == 1 else (min(x_test, x0), max(x0, x_test))
                    # Actually, we need to be more careful - use the two points that bracket
                    a = x0 if direction == 1 else x_test
                    b = x_test if direction == 1 else x0
                    fa = ref_f
                    fb = f_test
                    if fa * fb < 0:
                        try:
                            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                            if np.isfinite(root):
                                return root
                        except:
                            pass
                    break
                else:
                    # No sign change, update reference to current point
                    bracket = (x_test, f_test)
    
    # If no bracket found via sign change, try to find a minimum
    # Use golden section search in a reasonable range
    if not time_ok():
        return best_x
    
    # Try golden section search to find a local minimum of |f|
    # Search in a range around x0
    search_range = max(1.0, abs(x0)) * 10
    
    def golden_section_search(f, a, b, max_iter=50):
        """Find minimum of f in [a, b] using golden section search."""
        if not time_ok():
            return None
        
        gr = (math.sqrt(5) - 1) / 2  # golden ratio
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = f(c)
        fd = f(d)
        
        for _ in range(max_iter):
            if not time_ok():
                break
            
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = f(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = f(d)
        
        return (a + b) / 2
    
    # Try several ranges for golden section
    for mult in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
        if not time_ok():
            break
        
        a = x0 - search_range * mult
        b = x0 + search_range * mult
        
        if a >= b:
            continue
        
        x_min = golden_section_search(lambda x: abs(f(x)), a, b, max_iter=30)
        
        if x_min is not None and np.isfinite(x_min):
            f_min = f(x_min)
            if np.isfinite(f_min) and abs(f_min) < best_f:
                best_f = abs(f_min)
                best_x = x_min
            
            if abs(f_min) <= 1e-8:
                return x_min
    
    # Final attempt: try Newton's method from best_x if we haven't succeeded
    if not time_ok():
        return best_x
    
    x = best_x
    for _ in range(30):
        if not time_ok():
            break
        
        fx = f(x)
        if not np.isfinite(fx):
            break
        if abs(fx) <= 1e-8:
            return x
        
        h = 1e-7 * max(1.0, abs(x))
        d = (f(x + h) - fx) / h
        if d == 0 or not np.isfinite(d):
            break
        
        x_new = x - fx / d
        if not np.isfinite(x_new):
            break
        
        if abs(x_new - x) < 1e-12:
            break
        
        x = x_new
    
    return x
# EVOLVE-BLOCK-END
