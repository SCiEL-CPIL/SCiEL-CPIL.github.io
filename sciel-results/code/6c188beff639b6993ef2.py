import numpy as np
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0
    
    x0 = float(x0)
    
    def check_deadline():
        return time.time() > deadline
    
    # Strategy 1: Try polynomial reconstruction with 4 points (cubic)
    # Use adaptive spread around x0
    best_x = x0
    best_fabs = abs(f(x0))
    
    # Try several spreads to find a good interpolation window
    spreads = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    found = False
    for h in spreads:
        if check_deadline():
            break
        points = [x0 - 1.5*h, x0 - 0.5*h, x0 + 0.5*h, x0 + 1.5*h]
        try:
            vals = [f(p) for p in points]
            # Check for non-finite
            if any(not np.isfinite(v) for v in vals):
                continue
            
            # Fit cubic polynomial
            coeffs = np.polyfit(points, vals, 3)
            
            # Check fit quality: evaluate poly at points, compare
            poly = np.poly1d(coeffs)
            residuals = [abs(poly(p) - v) for p, v in zip(points, vals)]
            max_res = max(residuals)
            scale = max(1.0, max(abs(v) for v in vals))
            
            if max_res / scale < 1e-10:
                # Good fit - find roots
                roots = np.roots(coeffs)
                # Filter real roots
                real_roots = [r.real for r in roots if abs(r.imag) < 1e-10 and np.isfinite(r.real)]
                
                if real_roots:
                    # Pick root closest to x0
                    candidate = min(real_roots, key=lambda r: abs(r - x0))
                    fcand = f(candidate)
                    if np.isfinite(fcand) and abs(fcand) <= 1e-8:
                        return candidate
                    elif np.isfinite(fcand) and abs(fcand) < best_fabs:
                        best_x = candidate
                        best_fabs = abs(fcand)
                    # Polish with Newton
                    x = candidate
                    for _ in range(20):
                        if check_deadline():
                            break
                        fx = f(x)
                        if abs(fx) <= 1e-12:
                            return x
                        h2 = 1e-7 * max(1.0, abs(x))
                        d = (f(x + h2) - fx) / h2
                        if d == 0 or not np.isfinite(d):
                            break
                        x_new = x - fx / d
                        if not np.isfinite(x_new):
                            break
                        if abs(x_new - x) < 1e-14 * max(1.0, abs(x)):
                            break
                        x = x_new
                    fx = f(x)
                    if np.isfinite(fx) and abs(fx) <= 1e-8:
                        return x
                    if np.isfinite(fx) and abs(fx) < best_fabs:
                        best_x = x
                        best_fabs = abs(fx)
                    found = True
                    break
        except Exception:
            continue
    
    # If cubic didn't work, try quadratic (3 points)
    if not found:
        for h in spreads:
            if check_deadline():
                break
            points = [x0 - h, x0, x0 + h]
            try:
                vals = [f(p) for p in points]
                if any(not np.isfinite(v) for v in vals):
                    continue
                coeffs = np.polyfit(points, vals, 2)
                poly = np.poly1d(coeffs)
                residuals = [abs(poly(p) - v) for p, v in zip(points, vals)]
                max_res = max(residuals)
                scale = max(1.0, max(abs(v) for v in vals))
                if max_res / scale < 1e-10:
                    roots = np.roots(coeffs)
                    real_roots = [r.real for r in roots if abs(r.imag) < 1e-10 and np.isfinite(r.real)]
                    if real_roots:
                        candidate = min(real_roots, key=lambda r: abs(r - x0))
                        fcand = f(candidate)
                        if np.isfinite(fcand) and abs(fcand) <= 1e-8:
                            return candidate
                        if np.isfinite(fcand) and abs(fcand) < best_fabs:
                            best_x = candidate
                            best_fabs = abs(fcand)
                        # Polish
                        x = candidate
                        for _ in range(20):
                            if check_deadline():
                                break
                            fx = f(x)
                            if abs(fx) <= 1e-12:
                                return x
                            h2 = 1e-7 * max(1.0, abs(x))
                            d = (f(x + h2) - fx) / h2
                            if d == 0 or not np.isfinite(d):
                                break
                            x_new = x - fx / h2 if h2 != 0 else x
                            if not np.isfinite(x_new):
                                break
                            if abs(x_new - x) < 1e-14 * max(1.0, abs(x)):
                                break
                            x = x_new
                        fx = f(x)
                        if np.isfinite(fx) and abs(fx) <= 1e-8:
                            return x
                        if np.isfinite(fx) and abs(fx) < best_fabs:
                            best_x = x
                            best_fabs = abs(fx)
                        found = True
                        break
            except Exception:
                continue
    
    # Fallback: iterative Newton/bisection hybrid
    if not found:
        x = best_x
        # Try Newton from x0
        x = x0
        for _ in range(100):
            if check_deadline():
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
            if abs(x_new - x) < 1e-14 * max(1.0, abs(x)):
                break
            x = x_new
        fx = f(x)
        if np.isfinite(fx) and abs(fx) <= 1e-8:
            return x
        if np.isfinite(fx) and abs(fx) < best_fabs:
            best_x = x
            best_fabs = abs(fx)
    
    return best_x
# EVOLVE-BLOCK-END
