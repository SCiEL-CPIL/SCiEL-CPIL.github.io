"""Root finder for smooth polynomial equations - Muller's Method with safeguards."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    x = float(x0)
    
    # 1. Evaluate initial point
    try:
        fx = f(x)
    except Exception:
        return x0

    if not math.isfinite(fx):
        # Try to find a finite point nearby
        for dx in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]:
            for sign in [1, -1]:
                xt = x + sign * dx
                try:
                    fxt = f(xt)
                except Exception:
                    continue
                if math.isfinite(fxt):
                    x = xt
                    fx = fxt
                    break
            if math.isfinite(fx):
                break
        if not math.isfinite(fx):
            return x0

    if abs(fx) <= 1e-8:
        return x

    # 2. Find a second point
    # We need distinct x values with finite f values
    x1 = None
    fx1 = None
    best_x = x
    best_fx = fx
    best_abs = abs(fx)

    # Try small steps first to stay local
    h_base = max(1e-4, abs(x) * 1e-4)
    steps = [h_base, h_base*0.1, h_base*10, h_base*0.01, h_base*100, 1e-2, 0.1, 1.0, 10.0]
    
    for h in steps:
        for sign in [1, -1]:
            xt = x + sign * h
            if xt == x: continue
            try:
                fxt = f(xt)
            except Exception:
                continue
            if not math.isfinite(fxt):
                continue
            
            if abs(fxt) < best_abs:
                best_x = xt
                best_fx = fxt
                best_abs = abs(fxt)
            
            if abs(fxt) <= 1e-8:
                return xt
            
            # For Muller, we just need a valid second point.
            # Ideally, one where f changes sign or is significantly different.
            if abs(fxt - fx) > 1e-12:
                x1 = xt
                fx1 = fxt
                break
        if x1 is not None:
            break

    if x1 is None:
        # Couldn't find a distinct point with different f value?
        # This implies f is constant or we are in a weird flat region.
        # Return best guess.
        return best_x

    # Now we have (x, fx) and (x1, fx1)
    # We need a third point for Muller's method.
    # Let's use the secant method to get a third point, or just pick one.
    # Actually, we can start with Secant to get a third point, then switch to Muller.
    
    x0_curr = x
    f0_curr = fx
    x1_curr = x1
    f1_curr = fx1
    
    # We need a third point. Let's generate it via Secant step from the first two.
    # This effectively bootstraps the 3 points.
    
    x2 = None
    f2 = None
    
    denom = f1_curr - f0_curr
    if abs(denom) > 1e-14:
        step = f1_curr * (x1_curr - x0_curr) / denom
        x2 = x1_curr - step
        # Safeguard: don't go too far
        if abs(x2 - x1_curr) > max(1.0, abs(x1_curr)):
             x2 = x1_curr + math.copysign(max(1.0, abs(x1_curr)), -step)
        
        try:
            f2 = f(x2)
        except Exception:
            f2 = float('inf')
        
        if not math.isfinite(f2):
            # Fallback: try a small step
            x2 = x1_curr + 1e-3
            try:
                f2 = f(x2)
            except Exception:
                f2 = float('inf')
        
        if not math.isfinite(f2):
            # If we can't get a third point, just do secant iterations
            return _secant_fallback(f, x0_curr, f0_curr, x1_curr, f1_curr, best_x, best_fx, deadline)
        
        if abs(f2) <= 1e-8:
            return x2
    else:
        # Parallel lines, can't secant. Pick arbitrary 3rd point
        x2 = x1_curr + 1e-3
        try:
            f2 = f(x2)
        except Exception:
            f2 = float('inf')
        if not math.isfinite(f2):
            return best_x

    # Now we have 3 points: (x0_curr, f0_curr), (x1_curr, f1_curr), (x2, f2)
    # Sort them by x? Muller's method typically uses the last 3 points.
    # Let's keep them in a list and append new ones.
    
    pts = [(x0_curr, f0_curr), (x1_curr, f1_curr), (x2, f2)]
    
    # Update best
    if abs(f2) < best_abs:
        best_x = x2
        best_fx = f2
        best_abs = abs(f2)

    max_iter = 50
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        x_a, f_a = pts[-3]
        x_b, f_b = pts[-2]
        x_c, f_c = pts[-1]
        
        # Check convergence
        if abs(f_c) <= 1e-8:
            return x_c
        if abs(f_b) <= 1e-8:
            return x_b
        if abs(f_a) <= 1e-8:
            return x_a

        # Muller's Method Step
        # h1 = x_b - x_a, h2 = x_c - x_b
        h1 = x_b - x_a
        h2 = x_c - x_b
        
        # Avoid division by zero
        if abs(h1) < 1e-14 or abs(h2) < 1e-14:
             # Points are too close, perturb
             x_new = x_c + 1e-6
             try:
                 f_new = f(x_new)
             except Exception:
                 break
             if not math.isfinite(f_new):
                 break
             pts.append((x_new, f_new))
             continue

        # Divided differences
        d1 = (f_b - f_a) / h1
        d2 = (f_c - f_b) / h2
        d3 = (d2 - d1) / (h1 + h2)
        
        # Quadratic coefficients for the interpolating polynomial
        # f(x) = f_c + b*(x-x_c) + a*(x-x_c)^2
        # a = d3
        # b = d2 + d3*h2
        a = d3
        b = d2 + d3 * h2
        
        # Solve a*s^2 + b*s + f_c = 0 for s
        # s = -b/2a +/- sqrt(b^2/4a^2 - f_c/a)
        # Or use the stable form:
        # s = 2 * f_c / (b + sign(b) * sqrt(b^2 - 4*a*f_c))
        
        disc = b*b - 4*a*f_c
        
        if disc < 0:
            # Complex roots? For real polynomials, this can happen if the local quadratic
            # doesn't cross the axis. Fall back to Secant or Bisection.
            # Let's try Secant on the last two points
            denom = f_c - f_b
            if abs(denom) > 1e-14:
                s = f_c * (x_c - x_b) / denom
                x_new = x_c - s
            else:
                x_new = x_c + 1e-6
        else:
            sqrt_disc = math.sqrt(disc)
            # Choose the sign of the square root to avoid catastrophic cancellation
            # If b > 0, use b + sqrt_disc, else b - sqrt_disc
            if b >= 0:
                denom = b + sqrt_disc
            else:
                denom = b - sqrt_disc
            
            if abs(denom) < 1e-14:
                # Fallback
                denom = f_c - f_b
                if abs(denom) > 1e-14:
                    s = f_c * (x_c - x_b) / denom
                    x_new = x_c - s
                else:
                    x_new = x_c + 1e-6
            else:
                s = 2 * f_c / denom
                x_new = x_c + s
        
        # Safeguard: Ensure x_new is not too far away
        # If the step is huge, clamp it
        max_step = max(1.0, abs(x_c))
        if abs(x_new - x_c) > max_step:
            x_new = x_c + math.copysign(max_step, x_new - x_c)
            
        try:
            f_new = f(x_new)
        except Exception:
            break

        if not math.isfinite(f_new):
            # Shrink step
            x_new = x_c + 0.5 * (x_new - x_c)
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break

        if abs(f_new) < best_abs:
            best_x = x_new
            best_fx = f_new
            best_abs = abs(f_new)

        if abs(f_new) <= 1e-8:
            return x_new

        # Update points: drop the oldest, add the new one
        pts.pop(0)
        pts.append((x_new, f_new))

    return best_x

def _secant_fallback(f, x0, f0, x1, f1, best_x, best_fx, deadline):
    # Simple secant loop if Muller init failed
    x0_s, f0_s = x0, f0
    x1_s, f1_s = x1, f1
    max_iter = 30
    for i in range(max_iter):
        if time.time() > deadline:
            break
        if abs(f1_s) <= 1e-8:
            return x1_s
        denom = f1_s - f0_s
        if abs(denom) < 1e-14:
            break
        step = f1_s * (x1_s - x0_s) / denom
        x_new = x1_s - step
        try:
            f_new = f(x_new)
        except Exception:
            break
        if not math.isfinite(f_new):
            break
        if abs(f_new) < abs(best_fx):
            best_x = x_new
            best_fx = f_new
        if abs(f_new) <= 1e-8:
            return x_new
        x0_s, f0_s = x1_s, f1_s
        x1_s, f1_s = x_new, f_new
    return best_x

# EVOLVE-BLOCK-END
