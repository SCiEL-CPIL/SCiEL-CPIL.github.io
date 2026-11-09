import time
import numpy as np

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s kill
    
    def is_deadline_reached():
        return time.time() > deadline
    
    def is_finite(x):
        return x is not None and isinstance(x, (int, float)) and np.isfinite(x)
    
    if not is_finite(x0):
        x0 = 0.0
    x0 = float(x0)
    
    best_x = x0
    best_f_abs = float('inf')
    points = []
    
    def add_point(x, fx):
        nonlocal best_x, best_f_abs
        if not is_finite(x) or not is_finite(fx):
            return False
        for px, pfx in points:
            if abs(x - px) < 1e-13:
                return False
        points.append((x, fx))
        if abs(fx) < best_f_abs:
            best_f_abs = abs(fx)
            best_x = x
        return True
    
    # Evaluate initial point
    fx0 = f(x0)
    add_point(x0, fx0)
    if abs(fx0) <= 1e-8:
        return x0
    if is_deadline_reached():
        return best_x
    
    # Get a second point with adaptive step
    step = 1e-4 * max(1.0, abs(x0))
    x1 = x0 + step
    fx1 = f(x1)
    add_point(x1, fx1)
    if abs(fx1) <= 1e-8:
        return x1
    if is_deadline_reached():
        return best_x
    
    # If signs differ, we have a bracket - use Illinois/Regula Falsi
    if fx0 * fx1 < 0:
        a, b = x0, x1
        fa, fb = fx0, fx1
        
        for _ in range(50):
            if is_deadline_reached():
                break
            # Secant step within bracket
            if fb == fa:
                x_new = (a + b) / 2.0
            else:
                x_new = a - fa * (b - a) / (fb - fa)
            
            # If secant goes outside bracket, use bisection
            if x_new <= a or x_new >= b:
                x_new = (a + b) / 2.0
            
            fx_new = f(x_new)
            if not is_finite(fx_new):
                break
            add_point(x_new, fx_new)
            if abs(fx_new) <= 1e-8:
                return x_new
            
            if fa * fx_new < 0:
                b, fb = x_new, fx_new
            else:
                a, fa = x_new, fx_new
                # Illinois method: shrink the side we're keeping
                fa = fa / 2.0
    else:
        # No sign change - use Newton's method with numerical derivative
        if fx0 == fx1:
            x1 = x0 - step
            fx1 = f(x1)
            add_point(x1, fx1)
            if abs(fx1) <= 1e-8:
                return x1
            if fx0 == fx1:
                return best_x
        
        # Start Newton from the point with smallest |f|
        pts_sorted = sorted(points, key=lambda p: abs(p[1]))
        x_curr = pts_sorted[0][0]
        fx_curr = pts_sorted[0][1]
        h = 1e-6 * max(1.0, abs(x_curr))
        
        for _ in range(30):
            if is_deadline_reached():
                break
            
            # Numerical derivative using central difference
            fx_plus = f(x_curr + h)
            if not is_finite(fx_plus):
                break
            fx_minus = f(x_curr - h)
            if not is_finite(fx_minus):
                break
            
            dfx = (fx_plus - fx_minus) / (2 * h)
            
            if abs(dfx) < 1e-15:
                # Derivative too small, try secant
                if len(points) >= 2:
                    p1, p2 = pts_sorted[0], pts_sorted[1]
                    if p1[1] == p2[1]:
                        x_curr = x_curr + 1e-4
                    else:
                        x_curr = p1[0] - p1[1] * (p2[0] - p1[0]) / (p2[1] - p1[1])
                else:
                    x_curr = x_curr + 1e-4
                fx_curr = f(x_curr)
                add_point(x_curr, fx_curr)
                if abs(fx_curr) <= 1e-8:
                    return x_curr
                continue
            
            x_new = x_curr - fx_curr / dfx
            
            if not is_finite(x_new):
                break
            
            fx_new = f(x_new)
            if not is_finite(fx_new):
                break
            
            add_point(x_new, fx_new)
            if abs(fx_new) <= 1e-8:
                return x_new
            
            # Damping: if we moved too far, take a smaller step
            if abs(x_new - x_curr) > 10 * max(1.0, abs(x_curr)):
                x_curr = x_curr + 0.5 * (x_new - x_curr)
                fx_curr = f(x_curr)
                add_point(x_curr, fx_curr)
                if abs(fx_curr) <= 1e-8:
                    return x_curr
            else:
                x_curr = x_new
                fx_curr = fx_new
            
            # Update h
            h = max(1e-8, 1e-6 * max(1.0, abs(x_curr)))
    
    return best_x if is_finite(best_x) else x0
