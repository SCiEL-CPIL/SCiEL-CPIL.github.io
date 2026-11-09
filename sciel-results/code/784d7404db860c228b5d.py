"""Root finder for 3 fixed oscillatory equations with optimized call budget."""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x with |f(x)| <= 1e-8.
    
    Strategy:
    1. Dense uniform grid scan using NumPy to efficiently locate sign-change intervals.
    2. High-precision bisection/secant within the identified bracket.
    3. Final polish pass if needed.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    tol = 1e-8
    
    x0 = float(x0)
    
    # 1. Initial Evaluation
    try:
        f0 = f(x0)
    except Exception:
        x_try = x0 + 1e-6
        try:
            f0 = f(x_try)
            x0 = x_try
        except Exception:
            return x0
    
    if not math.isfinite(f0):
        return x0
        
    if abs(f0) <= tol:
        return x0

    # Track best point found
    best_x = x0
    best_f = f0
    best_abs = abs(f0)
    
    def update_best(x, fx):
        nonlocal best_x, best_f, best_abs
        if math.isfinite(x) and math.isfinite(fx):
            af = abs(fx)
            if af < best_abs:
                best_x = x
                best_f = fx
                best_abs = af

    # 2. Dense grid scan to find bracket
    # Use multiple scales: start local, expand if needed
    scales = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    n_points = 201  # Odd number for symmetric grid around x0
    
    a = None
    b = None
    fa = None
    fb = None
    
    for scale in scales:
        if time.time() > deadline:
            break
            
        # Create uniform grid centered at x0
        # Range: [x0 - scale, x0 + scale]
        xs = np.linspace(x0 - scale, x0 + scale, n_points)
        
        # Evaluate f on all grid points
        fs = np.empty(n_points)
        for i, xi in enumerate(xs):
            try:
                fi = f(float(xi))
            except Exception:
                fi = float('nan')
            fs[i] = fi
            if math.isfinite(fi):
                update_best(float(xi), fi)
                if abs(fi) <= tol:
                    return float(xi)
        
        # Check for sign changes
        finite_mask = np.isfinite(fs)
        if not np.any(finite_mask):
            continue
            
        # Find indices where sign changes occur
        # A sign change exists between i and i+1 if fs[i]*fs[i+1] < 0
        # and both are finite
        for i in range(n_points - 1):
            if not (finite_mask[i] and finite_mask[i+1]):
                continue
            if fs[i] * fs[i+1] < 0:
                a = float(xs[i])
                b = float(xs[i+1])
                fa = float(fs[i])
                fb = float(fs[i+1])
                break
        
        if a is not None:
            break

    # 3. If bracket found, use high-precision bisection
    if a is not None and b is not None:
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
            
        # Bisection with secant acceleration
        for _ in range(60):
            if time.time() > deadline:
                break
                
            if abs(fa) <= tol: 
                return a
            if abs(fb) <= tol: 
                return b
            
            # Try secant first
            denom = fb - fa
            if denom != 0 and math.isfinite(denom):
                x_secant = b - fb * (b - a) / denom
                if a < x_secant < b:
                    x_new = x_secant
                else:
                    x_new = (a + b) / 2
            else:
                x_new = (a + b) / 2
                
            if not math.isfinite(x_new):
                break
                
            try:
                f_new = f(x_new)
            except Exception:
                break
                
            if not math.isfinite(f_new):
                break
                
            update_best(x_new, f_new)
            
            if abs(f_new) <= tol:
                return x_new
                
            # Update bracket
            if fa * f_new < 0:
                b, fb = x_new, f_new
            else:
                a, fa = x_new, f_new
                
            # Convergence check
            if abs(b - a) < 1e-15 * max(1.0, abs(b)):
                break
        
        # Polish pass
        if best_abs <= 1e-6:
            if a is not None and b is not None and min(a,b) <= best_x <= max(a,b):
                pa, pb = a, b
                pfa, pfb = fa, fb
            else:
                eps = 1e-10
                pa, pb = best_x - eps, best_x + eps
                try:
                    pfa = f(pa)
                    pfb = f(pb)
                except Exception:
                    pfa, pfb = float('nan'), float('nan')
                
                if not (math.isfinite(pfa) and math.isfinite(pfb)):
                    return best_x

            if math.isfinite(pfa) and math.isfinite(pfb) and pfa * pfb < 0:
                for _ in range(25):
                    if time.time() > deadline:
                        break
                    if abs(pfa) <= tol: 
                        return pa
                    if abs(pfb) <= tol: 
                        return pb
                    
                    x_mid = (pa + pb) / 2
                    try:
                        f_mid = f(x_mid)
                    except Exception:
                        break
                    if not math.isfinite(f_mid):
                        break
                    update_best(x_mid, f_mid)
                    if abs(f_mid) <= tol:
                        return x_mid
                    
                    if pfa * f_mid < 0:
                        pb, pfb = x_mid, f_mid
                    else:
                        pa, pfa = x_mid, f_mid

        return best_x if best_abs <= tol else (a + b) / 2

    else:
        # No bracket found - fallback to secant
        x1 = x0
        f1 = f0
        
        # Find second point
        candidates = [x0 + 1e-2, x0 - 1e-2, x0 + 1.0, x0 - 1.0, x0 + 10.0]
        x2 = None
        f2 = None
        min_abs = float('inf')
        
        for cx in candidates:
            try:
                cf = f(cx)
            except Exception:
                continue
            if math.isfinite(cf):
                update_best(cx, cf)
                if abs(cf) < min_abs:
                    min_abs = abs(cf)
                    x2 = cx
                    f2 = cf
        
        if x2 is None:
            return best_x

        # Secant iteration
        for _ in range(30):
            if time.time() > deadline:
                break
                
            if abs(f1) <= tol: 
                return x1
            if abs(f2) <= tol: 
                return x2
            
            denom = f2 - f1
            if denom == 0 or not math.isfinite(denom):
                break
                
            x_new = x2 - f2 * (x2 - x1) / denom
            
            if not math.isfinite(x_new):
                break
                
            try:
                f_new = f(x_new)
            except Exception:
                break
                
            if not math.isfinite(f_new):
                break
                
            update_best(x_new, f_new)
            
            if abs(f_new) <= tol:
                return x_new
                
            x1, f1 = x2, f2
            x2, f2 = x_new, f_new
            
        # Polish pass
        if best_abs <= 1e-6:
            eps = 1e-10
            pa, pb = best_x - eps, best_x + eps
            try:
                pfa = f(pa)
                pfb = f(pb)
            except Exception:
                return best_x
             
            if math.isfinite(pfa) and math.isfinite(pfb) and pfa * pfb < 0:
                for _ in range(25):
                    if time.time() > deadline:
                        break
                    if abs(pfa) <= tol: 
                        return pa
                    if abs(pfb) <= tol: 
                        return pb
                     
                    x_mid = (pa + pb) / 2
                    try:
                        f_mid = f(x_mid)
                    except Exception:
                        break
                    if not math.isfinite(f_mid):
                        break
                    update_best(x_mid, f_mid)
                    if abs(f_mid) <= tol:
                        return x_mid
                     
                    if pfa * f_mid < 0:
                        pb, pfb = x_mid, f_mid
                    else:
                        pa, pfa = x_mid, f_mid

        return best_x

# EVOLVE-BLOCK-END
