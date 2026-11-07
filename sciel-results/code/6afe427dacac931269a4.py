"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import brentq

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 135.0  # safety margin before 150s kill
    x0 = float(x0)
    
    call_count = 0
    def time_ok():
        nonlocal call_count
        return (time.time() - start_time) < TIME_LIMIT and call_count < 480
    
    # Check initial point
    f0 = f(x0)
    call_count += 1
    if not np.isfinite(f0):
        f0 = float('inf')
    if abs(f0) <= 1e-8:
        return x0
    
    best_x = x0
    best_f = abs(f0)
    
    def try_brentq(a, b):
        nonlocal best_x, best_f, call_count
        try:
            fa = f(a)
            call_count += 1
            fb = f(b)
            call_count += 1
            if not np.isfinite(fa) or not np.isfinite(fb):
                return None
            if abs(fa) < best_f:
                best_f = abs(fa)
                best_x = a
            if abs(fb) < best_f:
                best_f = abs(fb)
                best_x = b
            if fa == 0:
                return a
            if fb == 0:
                return b
            if fa * fb > 0:
                return None
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
            call_count += 1  # approximate; brentq makes several calls
            if np.isfinite(root):
                fr = f(root)
                call_count += 1
                if abs(fr) <= 1e-8:
                    return root
        except Exception:
            pass
        return None
    
    # Multi-scale sign-change scan
    # Try various scales to find sign changes near x0
    scales = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    
    for scale in scales:
        if not time_ok():
            break
        
        # Scan in both directions from x0
        for direction in [-1, 1]:
            if not time_ok():
                break
            
            prev_x = x0
            prev_f = f0
            
            # Scan up to 10 steps in each direction at this scale
            for i in range(1, 11):
                if not time_ok():
                    break
                
                x = x0 + direction * i * scale
                fx = f(x)
                call_count += 1
                
                if not np.isfinite(fx):
                    break
                
                if abs(fx) < best_f:
                    best_f = abs(fx)
                    best_x = x
                
                # Check for sign change
                if prev_f * fx < 0:
                    root = try_brentq(prev_x, x)
                    if root is not None:
                        return root
                
                prev_x = x
                prev_f = fx
    
    # If no sign change found, try a denser local scan
    if time_ok():
        local_scales = [0.001, 0.005, 0.02, 0.08, 0.2, 0.8]
        for scale in local_scales:
            if not time_ok():
                break
            
            for direction in [-1, 1]:
                if not time_ok():
                    break
                
                prev_x = x0
                prev_f = f0
                
                for i in range(1, 21):
                    if not time_ok():
                        break
                    
                    x = x0 + direction * i * scale
                    fx = f(x)
                    call_count += 1
                    
                    if not np.isfinite(fx):
                        break
                    
                    if abs(fx) < best_f:
                        best_f = abs(fx)
                        best_x = x
                    
                    if prev_f * fx < 0:
                        root = try_brentq(prev_x, x)
                        if root is not None:
                            return root
                    
                    prev_x = x
                    prev_f = fx
    
    # Return best x found
    return best_x
# EVOLVE-BLOCK-END
