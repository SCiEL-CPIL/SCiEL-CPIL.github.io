"""Single-shot cubic reconstruction via adaptive point selection."""
import numpy as np

def find_root(f, x0):
    x = float(x0)
    
    # Step 1: Check if x0 is already the root
    try:
        y0 = f(x)
    except Exception:
        return x
        
    if abs(y0) <= 1e-8:
        return x

    # Step 2: Adaptive point selection for cubic interpolation
    # Use multiple scale factors to find a good bracket
    # Try several step sizes to capture the root well
    scales = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    
    best_root = x
    best_y = y0
    calls_used = 1  # y0 already counted
    
    for scale in scales:
        h = scale * max(1.0, abs(x))
        if h == 0:
            h = 1e-10
        
        # Use 4 points: x-h, x, x+h, x+2h for asymmetric coverage
        p1 = x - h
        p2 = x
        p3 = x + h
        p4 = x + 2 * h
        
        try:
            y_p1 = f(p1)
            y_p3 = f(p3)
            y_p4 = f(p4)
        except Exception:
            continue
        
        calls_used += 3
        
        # Try quadratic fit first (cheaper, more stable)
        try:
            xs_q = np.array([p1, p2, p3], dtype=np.float64)
            ys_q = np.array([y_p1, y0, y_p3], dtype=np.float64)
            coeffs_q = np.polyfit(xs_q, ys_q, 2)
            roots_q = np.roots(coeffs_q)
            real_roots_q = [r.real for r in roots_q if abs(r.imag) < 1e-10]
            
            if real_roots_q:
                x_q = min(real_roots_q, key=lambda r: abs(r - x))
                y_q = f(x_q)
                calls_used += 1
                if abs(y_q) < abs(best_y):
                    best_root = x_q
                    best_y = y_q
                if abs(y_q) <= 1e-8:
                    return x_q
        except Exception:
            pass
        
        # Try cubic fit
        try:
            xs_c = np.array([p1, p2, p3, p4], dtype=np.float64)
            ys_c = np.array([y_p1, y0, y_p3, y_p4], dtype=np.float64)
            coeffs_c = np.polyfit(xs_c, ys_c, 3)
            roots_c = np.roots(coeffs_c)
            real_roots_c = [r.real for r in roots_c if abs(r.imag) < 1e-10]
            
            if real_roots_c:
                x_c = min(real_roots_c, key=lambda r: abs(r - x))
                y_c = f(x_c)
                calls_used += 1
                if abs(y_c) < abs(best_y):
                    best_root = x_c
                    best_y = y_c
                if abs(y_c) <= 1e-8:
                    return x_c
        except Exception:
            pass
        
        # If we found a good root, stop early
        if abs(best_y) <= 1e-6:
            break
    
    # Step 3: If best root is good enough, return it
    if abs(best_y) <= 1e-8:
        return best_root
    
    # Step 4: Use Newton's method from the best candidate found
    x = best_root
    for _ in range(20):
        try:
            fx = f(x)
            if abs(fx) <= 1e-8:
                return x
            h = 1e-6 * max(1.0, abs(x))
            d = (f(x + h) - fx) / h
            if d == 0:
                break
            x_new = x - fx / d
            # Damping: take half step if it's too large
            if abs(x_new - x) > 0.1 * max(1.0, abs(x)):
                x_new = x - 0.5 * (x_new - x)
            x = x_new
        except Exception:
            break
    
    return x
