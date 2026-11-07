import numpy as np
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0
    
    x0 = float(x0)
    
    def check_deadline():
        return time.time() > deadline

    # Scale h based on x0 to handle large/small x0
    h = 1e-3 * max(1.0, abs(x0))
    
    # Points
    x1 = x0 - h
    x2 = x0
    x3 = x0 + h
    x4 = x0 + 2*h
    
    f1 = f2 = f3 = f4 = None
    
    # 1. Try Linear (x2, x3)
    try:
        f2 = f(x2)
        if not np.isfinite(f2):
            return x0
        if abs(f2) <= 1e-8: return x2
        if check_deadline(): return x2
        
        f3 = f(x3)
        if not np.isfinite(f3):
            return x2
        if abs(f3) <= 1e-8: return x3
        if check_deadline(): return x2
        
        if abs(f3 - f2) > 1e-15:
            x_lin = x2 - f2 * (x3 - x2) / (f3 - f2)
            if np.isfinite(x_lin):
                fx = f(x_lin)
                if np.isfinite(fx) and abs(fx) <= 1e-8:
                    return x_lin
    except Exception:
        pass

    # 2. Try Quadratic (x1, x2, x3)
    try:
        f1 = f(x1)
        if not np.isfinite(f1):
            f1 = None
        else:
            if abs(f1) <= 1e-8: return x1
            if check_deadline(): return x2
            
            if f1 is not None and f2 is not None and f3 is not None:
                # Solve quadratic ax^2 + bx + c = 0
                # Using np.polyfit is robust enough for 3 points
                coeffs = np.polyfit([x1, x2, x3], [f1, f2, f3], 2)
                roots = np.roots(coeffs)
                real_roots = [r.real for r in roots if abs(r.imag) < 1e-10]
                if real_roots:
                    best_root = min(real_roots, key=lambda r: abs(r - x2))
                    if np.isfinite(best_root):
                        fx = f(best_root)
                        if np.isfinite(fx) and abs(fx) <= 1e-8:
                            return best_root
    except Exception:
        pass

    # 3. Try Cubic (x1, x2, x3, x4)
    try:
        if f1 is None:
            f1 = f(x1)
            if not np.isfinite(f1):
                f1 = None
            else:
                if abs(f1) <= 1e-8: return x1
                
        if f1 is not None and f2 is not None and f3 is not None:
            f4 = f(x4)
            if not np.isfinite(f4):
                f4 = None
            else:
                if abs(f4) <= 1e-8: return x4
                if check_deadline(): return x2
                
                if f4 is not None:
                    coeffs = np.polyfit([x1, x2, x3, x4], [f1, f2, f3, f4], 3)
                    roots = np.roots(coeffs)
                    real_roots = [r.real for r in roots if abs(r.imag) < 1e-10]
                    if real_roots:
                        best_root = min(real_roots, key=lambda r: abs(r - x2))
                        if np.isfinite(best_root):
                            fx = f(best_root)
                            if np.isfinite(fx) and abs(fx) <= 1e-8:
                                return best_root
    except Exception:
        pass

    # 4. Fallback: Secant Method
    # Ensure we have two valid points
    # Prefer points closest to x0 with valid f
    candidates = []
    if f1 is not None and np.isfinite(f1): candidates.append((x1, f1))
    if f2 is not None and np.isfinite(f2): candidates.append((x2, f2))
    if f3 is not None and np.isfinite(f3): candidates.append((x3, f3))
    if f4 is not None and np.isfinite(f4): candidates.append((x4, f4))
    
    if len(candidates) < 2:
        # Generate more points if needed
        # Try x0 - 2h, x0 + 4h etc?
        # Just return best guess
        if candidates:
            return candidates[0][0]
        return x0

    # Sort by closeness to x0? Or just take first two distinct?
    # Take x2, x3 if available, else others
    # Let's just take the first two valid ones
    xa, fa = candidates[0]
    xb, fb = candidates[1]
    
    if abs(fa - fb) < 1e-15:
        # Find another pair
        for i in range(len(candidates)):
            for j in range(i+1, len(candidates)):
                if abs(candidates[i][1] - candidates[j][1]) > 1e-15:
                    xa, fa = candidates[i]
                    xb, fb = candidates[j]
                    break
            if abs(fa - fb) > 1e-15: break
        
    if abs(fa - fb) < 1e-15:
        return xa if abs(fa) < abs(fb) else xb

    for _ in range(50):
        if check_deadline():
            return xa if abs(fa) < abs(fb) else xb
            
        denom = fb - fa
        if abs(denom) < 1e-15:
            break
        xc = xb - fb * (xb - xa) / denom
        if not np.isfinite(xc):
            break
        fc = f(xc)
        if not np.isfinite(fc):
            break
            
        if abs(fc) <= 1e-8:
            return xc
            
        xa, fa = xb, fb
        xb, fb = xc, fc
        
    return xa if abs(fa) < abs(fb) else xb
