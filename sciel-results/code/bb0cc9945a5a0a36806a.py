"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    # Deadline guard: kill after ~115s to stay under 120s wall-clock
    t_start = time.time()
    deadline = 115.0
    
    eval_count = 0
    max_evals = 500
    
    def f_counted(x):
        nonlocal eval_count
        eval_count += 1
        if eval_count > max_evals:
            raise ValueError("Max evaluations exceeded")
        return f(x)
    
    x = float(x0)
    
    # First evaluation
    fx = f_counted(x)
    best_x = x
    best_fx = fx
    best_abs_fx = abs(fx)
    
    if best_abs_fx <= 1e-8:
        return x
    
    # Get two more points for derivative estimation and IQI
    h = 1e-4 * max(1.0, abs(x))
    x1 = x + h
    x2 = x - h
    f1 = f_counted(x1)
    f2 = f_counted(x2)
    
    if abs(f1) <= 1e-8:
        return x1
    if abs(f2) <= 1e-8:
        return x2
    
    # Track best
    if abs(f1) < best_abs_fx:
        best_x = x1
        best_abs_fx = abs(f1)
    if abs(f2) < best_abs_fx:
        best_x = x2
        best_abs_fx = abs(f2)
    
    # Maintain three points for IQI / secant
    # Sort by |f| to keep the best three
    points = [(x, fx), (x1, f1), (x2, f2)]
    points.sort(key=lambda p: abs(p[1]))
    
    # Main loop: combine Newton (secant derivative) with IQI
    for i in range(max_evals - 3):
        # Check time
        if time.time() - t_start > deadline:
            break
        
        # Check if we've converged
        for px, pf in points:
            if abs(pf) <= 1e-8:
                return px
        
        # Extract the three points (sorted by |f|)
        x0_s, f0_s = points[0]
        x1_s, f1_s = points[1]
        x2_s, f2_s = points[2]
        
        x_new = None
        
        # Strategy 1: Newton's method with secant derivative
        # Use the two points closest to x0_s (best point) for derivative estimate
        # Pick the two points nearest to x0_s in x-space
        all_pts = points[:]
        all_pts.sort(key=lambda p: abs(p[0] - x0_s))
        px1, pf1 = all_pts[0]
        px2, pf2 = all_pts[1]
        
        if abs(px2 - px1) > 1e-15:
            df = (pf2 - pf1) / (px2 - px1)
            if abs(df) > 1e-30:
                x_newton = px1 - pf1 / df
                if math.isfinite(x_newton):
                    # Damping: if Newton step is too large, damp it
                    step = x_newton - px1
                    max_step = 5.0 * max(1.0, abs(px1))
                    if abs(step) > max_step:
                        x_newton = px1 + max_step * math.copysign(1.0, step)
                    x_new = x_newton
        
        # Strategy 2: IQI (Inverse Quadratic Interpolation)
        if x_new is None or not math.isfinite(x_new):
            if abs(x1_s - x0_s) > 1e-15 and abs(x2_s - x0_s) > 1e-15 and abs(x2_s - x1_s) > 1e-15:
                a = f0_s / ((x0_s - x1_s) * (x0_s - x2_s))
                b = f1_s / ((x1_s - x0_s) * (x1_s - x2_s))
                c = f2_s / ((x2_s - x0_s) * (x2_s - x1_s))
                
                A = a + b + c
                B = -(a * (x1_s + x2_s) + b * (x0_s + x2_s) + c * (x0_s + x1_s))
                C = a * x1_s * x2_s + b * x0_s * x2_s + c * x0_s * x1_s
                
                if abs(A) > 1e-15:
                    disc = B * B - 4 * A * C
                    if disc >= 0:
                        sqrt_disc = math.sqrt(disc)
                        x_cand1 = (-B + sqrt_disc) / (2 * A)
                        x_cand2 = (-B - sqrt_disc) / (2 * A)
                        
                        candidates = []
                        if math.isfinite(x_cand1):
                            candidates.append(x_cand1)
                        if math.isfinite(x_cand2):
                            candidates.append(x_cand2)
                        
                        if candidates:
                            # Choose candidate closest to best point
                            x_iqi = min(candidates, key=lambda xc: abs(xc - x0_s))
                            if x_new is None or not math.isfinite(x_new):
                                x_new = x_iqi
                            else:
                                # Prefer IQI if Newton step was large
                                if abs(x_new - x0_s) > 2.0 * max(1.0, abs(x0_s)):
                                    x_new = x_iqi
                elif abs(B) > 1e-15:
                    x_iqi = -C / B
                    if math.isfinite(x_iqi):
                        if x_new is None or not math.isfinite(x_new):
                            x_new = x_iqi
        
        # Strategy 3: Secant method on the two best points
        if x_new is None or not math.isfinite(x_new):
            if abs(f1_s - f0_s) > 1e-30:
                x_secant = x0_s - f0_s * (x1_s - x0_s) / (f1_s - f0_s)
                if math.isfinite(x_secant):
                    x_new = x_secant
        
        # Strategy 4: Bisection if sign change exists
        if x_new is None or not math.isfinite(x_new) or abs(x_new - x0_s) > 10 * max(1.0, abs(x0_s)):
            sorted_points = sorted(points, key=lambda p: p[0])
            (xa, fa), (xb, fb), (xc, fc) = sorted_points
            
            if fa * fb < 0:
                x_new = (xa + xb) / 2
            elif fb * fc < 0:
                x_new = (xb + xc) / 2
            elif fa * fc < 0:
                x_new = (xa + xc) / 2
            else:
                # No sign change; inverse linear interpolation
                p1, p2 = points[0], points[1]
                if abs(p2[1] - p1[1]) > 1e-30:
                    x_new = p1[0] - p1[1] * (p2[0] - p1[0]) / (p2[1] - p1[1])
                else:
                    x_new = x0_s + 1e-6 * max(1.0, abs(x0_s))
        
        # Check for NaN/Inf
        if not math.isfinite(x_new):
            break
        
        # Evaluate
        try:
            f_new = f_counted(x_new)
        except ValueError:
            break
        
        if not math.isfinite(f_new):
            break
        
        # Update best
        if abs(f_new) < best_abs_fx:
            best_x = x_new
            best_abs_fx = abs(f_new)
            if best_abs_fx <= 1e-8:
                return x_new
        
        # Update points: keep the three points with smallest |f|
        points.append((x_new, f_new))
        points.sort(key=lambda p: abs(p[1]))
        points = points[:3]
        
        # Check convergence
        for px, pf in points:
            if abs(pf) <= 1e-8:
                return px
    
    # If main loop didn't converge, try golden section search on |f(x)|
    if time.time() - t_start < deadline - 5.0 and eval_count < max_evals - 10:
        # Golden section search on |f(x)| in a window around best_x
        width = 1e-2 * max(1.0, abs(best_x))
        a = best_x - width
        b = best_x + width
        
        try:
            fa = f_counted(a)
            fb = f_counted(b)
        except ValueError:
            return best_x
        
        if abs(fa) <= 1e-8:
            return a
        if abs(fb) <= 1e-8:
            return b
        
        if abs(fa) < best_abs_fx:
            best_x = a
            best_abs_fx = abs(fa)
        if abs(fb) < best_abs_fx:
            best_x = b
            best_abs_fx = abs(fb)
        
        gr = (math.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        try:
            fc = f_counted(c)
            fd = f_counted(d)
        except ValueError:
            return best_x
        
        if abs(fc) < best_abs_fx:
            best_x = c
            best_abs_fx = abs(fc)
        if abs(fd) < best_abs_fx:
            best_x = d
            best_abs_fx = abs(fd)
        
        for _ in range(30):
            if time.time() - t_start > deadline - 2.0:
                break
            if eval_count >= max_evals - 5:
                break
            
            if abs(fc) < abs(fd):
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                try:
                    fc = f_counted(c)
                except ValueError:
                    break
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                try:
                    fd = f_counted(d)
                except ValueError:
                    break
            
            if abs(fc) < best_abs_fx:
                best_x = c
                best_abs_fx = abs(fc)
            if abs(fd) < best_abs_fx:
                best_x = d
                best_abs_fx = abs(fd)
            
            if best_abs_fx <= 1e-8:
                return best_x
    
    return best_x
# EVOLVE-BLOCK-END
