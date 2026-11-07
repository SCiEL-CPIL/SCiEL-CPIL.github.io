import time

# EVOLVE-BLOCK-START
import math

def find_root(f, x0):
    start_time = time.time()
    DEADLINE = 120.0  # wall-clock limit; safety margin below
    x = float(x0)
    
    # Track best point seen so far (smallest |f|)
    best_x = x
    best_fx = f(x)
    best_abs = abs(best_fx)
    
    def deadline_exceeded():
        return (time.time() - start_time) > (DEADLINE - 2.0)
    
    # If already close enough, return
    if not math.isfinite(best_abs):
        # f(x0) is non-finite; try to find a finite point
        for i in range(20):
            if deadline_exceeded():
                return best_x
            x_try = x0 + (i + 1) * 0.1
            fx_try = f(x_try)
            if math.isfinite(fx_try):
                best_x = x_try
                best_fx = fx_try
                best_abs = abs(fx_try)
                break
        return best_x
    
    if best_abs <= 1e-8:
        return best_x
    
    # Step 1: Find a bracket [a, b] where f(a) and f(b) have opposite signs
    # Start from x0 and expand
    a = x
    fa = best_fx
    step = 1.0
    b = None
    fb = None
    
    # Try to find a sign change by expanding in both directions
    for direction in [1, -1]:
        if deadline_exceeded():
            return best_x
        for i in range(60):
            if deadline_exceeded():
                return best_x
            b_try = a + direction * step * (1.0 + i)
            fb_try = f(b_try)
            if not math.isfinite(fb_try):
                continue
            # Update best
            if abs(fb_try) < best_abs:
                best_x = b_try
                best_fx = fb_try
                best_abs = abs(fb_try)
            if best_abs <= 1e-8:
                return best_x
            if fa * fb_try < 0:
                b = b_try
                fb = fb_try
                break
            if abs(fb_try) <= 1e-12:
                return b_try
        if b is not None:
            break
    
    # If we found a bracket, use IQI with bisection fallback
    if b is not None:
        # Ensure a and b are ordered
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
        
        # IQI: need three points. Start with a, b, and a midpoint or x0
        # Use a, b, and a point between them
        # We'll maintain three points: p1, p2, p3 with values
        # Start with a, b, and a third point
        c = (a + b) / 2.0
        fc = f(c)
        if not math.isfinite(fc):
            # Fallback to bisection only
            c = (a + b) / 2.0
            fc = f(c)
        
        # Maintain three most recent points for IQI
        # Points: (x1, f1), (x2, f2), (x3, f3)
        # We want to find root of quadratic through these three points
        
        # Initialize three points
        pts = [(a, fa), (c, fc), (b, fb)]
        
        for it in range(50):
            if deadline_exceeded():
                return best_x
            
            x1, f1 = pts[0]
            x2, f2 = pts[1]
            x3, f3 = pts[2]
            
            # Check if any point is already good enough
            for (xi, fi) in pts:
                if abs(fi) <= 1e-8:
                    # Polish
                    return _polish(f, xi, deadline_exceeded)
            
            # Update best
            for (xi, fi) in pts:
                if abs(fi) < best_abs:
                    best_x = xi
                    best_fx = fi
                    best_abs = abs(fi)
            
            # Inverse quadratic interpolation
            # Fit quadratic q(x) = A x^2 + B x + C through (x1,f1), (x2,f2), (x3,f3)
            # Solve for A, B, C
            # [x1^2 x1 1] [A]   [f1]
            # [x2^2 x2 1] [B] = [f2]
            # [x3^2 x3 1] [C]   [f3]
            
            # Use Cramer's rule or direct formula
            # det = x1^2*(x2 - x3) + x2^2*(x3 - x1) + x3^2*(x1 - x2)
            det = x1*x1*(x2 - x3) + x2*x2*(x3 - x1) + x3*x3*(x1 - x2)
            
            if abs(det) < 1e-30:
                # Degenerate, use bisection
                mid = (a + b) / 2.0
                fmid = f(mid)
                if not math.isfinite(fmid):
                    mid = (a + b) / 2.0
                    fmid = f(mid)
                if fa * fmid <= 0:
                    b = mid
                    fb = fmid
                else:
                    a = mid
                    fa = fmid
                # Update pts
                pts = [(a, fa), (mid, fmid), (b, fb)]
                continue
            
            # A = (f1*(x2-x3) + f2*(x3-x1) + f3*(x1-x2)) / det
            A = (f1*(x2 - x3) + f2*(x3 - x1) + f3*(x1 - x2)) / det
            # B = -(x1^2*(f2-f3) + x2^2*(f3-f1) + x3^2*(f1-f2)) / det
            B = -(x1*x1*(f2 - f3) + x2*x2*(f3 - f1) + x3*x3*(f1 - f2)) / det
            # C = (x1^2*x2*f3 - x1^2*x3*f2 - x2^2*x1*f3 + x2^2*x3*f1 + x3^2*x1*f2 - x3^2*x2*f1) / det
            C = (x1*x1*x2*f3 - x1*x1*x3*f2 - x2*x2*x1*f3 + x2*x2*x3*f1 + x3*x3*x1*f2 - x3*x3*x2*f1) / det
            
            if abs(A) < 1e-30:
                # Linear, just use linear interpolation
                # Root of B*x + C = 0
                x_new = -C / B if abs(B) > 1e-30 else (a + b) / 2.0
            else:
                # Quadratic: A*x^2 + B*x + C = 0
                disc = B*B - 4*A*C
                if disc < 0:
                    # No real roots, use bisection
                    x_new = (a + b) / 2.0
                else:
                    sqrt_disc = math.sqrt(disc)
                    r1 = (-B + sqrt_disc) / (2*A)
                    r2 = (-B - sqrt_disc) / (2*A)
                    # Choose the root that is within [a, b] or closer to the bracket
                    # Prefer root inside bracket
                    if a <= r1 <= b:
                        x_new = r1
                    elif a <= r2 <= b:
                        x_new = r2
                    else:
                        # Choose the one closer to the center of the bracket
                        mid = (a + b) / 2.0
                        if abs(r1 - mid) < abs(r2 - mid):
                            x_new = r1
                        else:
                            x_new = r2
            
            # Check if x_new is within the bracket
            if not (a <= x_new <= b):
                # Fall back to bisection
                x_new = (a + b) / 2.0
            
            fx_new = f(x_new)
            if not math.isfinite(fx_new):
                x_new = (a + b) / 2.0
                fx_new = f(x_new)
            
            if abs(fx_new) <= 1e-8:
                return _polish(f, x_new, deadline_exceeded)
            
            # Update bracket
            if fa * fx_new <= 0:
                b = x_new
                fb = fx_new
            else:
                a = x_new
                fa = fx_new
            
            # Update pts: keep the three most "diverse" points
            # Simple approach: replace the point farthest from x_new
            pts = [(a, fa), (x_new, fx_new), (b, fb)]
            
            # Check convergence of bracket
            if abs(b - a) < 1e-15 * max(1.0, abs(x_new)):
                # Return the point with smallest |f|
                candidates = [(a, fa), (x_new, fx_new), (b, fb)]
                best_local = min(candidates, key=lambda p: abs(p[1]))
                return _polish(f, best_local[0], deadline_exceeded)
        
        # Return best found
        return _polish(f, best_x, deadline_exceeded)
    
    else:
        # No bracket found. Try IQI without bracketing, or just return best
        # Try a few more points to find a sign change
        for i in range(30):
            if deadline_exceeded():
                return best_x
            x_try = x0 + (i + 1) * 10.0 * (1 if i % 2 == 0 else -1)
            fx_try = f(x_try)
            if not math.isfinite(fx_try):
                continue
            if abs(fx_try) < best_abs:
                best_x = x_try
                best_fx = fx_try
                best_abs = abs(fx_try)
            if best_abs <= 1e-8:
                return best_x
            # Check for sign change with best_x
            if best_fx * fx_try < 0:
                a = best_x
                fa = best_fx
                b = x_try
                fb = fx_try
                if a > b:
                    a, b = b, a
                    fa, fb = fb, fa
                # Now use IQI with this bracket
                c = (a + b) / 2.0
                fc = f(c)
                if not math.isfinite(fc):
                    fc = f((a+b)/2.0)
                pts = [(a, fa), (c, fc), (b, fb)]
                for it in range(50):
                    if deadline_exceeded():
                        return best_x
                    x1, f1 = pts[0]
                    x2, f2 = pts[1]
                    x3, f3 = pts[2]
                    for (xi, fi) in pts:
                        if abs(fi) <= 1e-8:
                            return _polish(f, xi, deadline_exceeded)
                    det = x1*x1*(x2 - x3) + x2*x2*(x3 - x1) + x3*x3*(x1 - x2)
                    if abs(det) < 1e-30:
                        mid = (a + b) / 2.0
                        fmid = f(mid)
                        if fa * fmid <= 0:
                            b = mid
                            fb = fmid
                        else:
                            a = mid
                            fa = fmid
                        pts = [(a, fa), (mid, fmid), (b, fb)]
                        continue
                    A = (f1*(x2 - x3) + f2*(x3 - x1) + f3*(x1 - x2)) / det
                    B = -(x1*x1*(f2 - f3) + x2*x2*(f3 - f1) + x3*x3*(f1 - f2)) / det
                    C = (x1*x1*x2*f3 - x1*x1*x3*f2 - x2*x2*x1*f3 + x2*x2*x3*f1 + x3*x3*x1*f2 - x3*x3*x2*f1) / det
                    if abs(A) < 1e-30:
                        x_new = -C / B if abs(B) > 1e-30 else (a + b) / 2.0
                    else:
                        disc = B*B - 4*A*C
                        if disc < 0:
                            x_new = (a + b) / 2.0
                        else:
                            sqrt_disc = math.sqrt(disc)
                            r1 = (-B + sqrt_disc) / (2*A)
                            r2 = (-B - sqrt_disc) / (2*A)
                            if a <= r1 <= b:
                                x_new = r1
                            elif a <= r2 <= b:
                                x_new = r2
                            else:
                                mid = (a + b) / 2.0
                                x_new = r1 if abs(r1 - mid) < abs(r2 - mid) else r2
                    if not (a <= x_new <= b):
                        x_new = (a + b) / 2.0
                    fx_new = f(x_new)
                    if not math.isfinite(fx_new):
                        x_new = (a + b) / 2.0
                        fx_new = f(x_new)
                    if abs(fx_new) <= 1e-8:
                        return _polish(f, x_new, deadline_exceeded)
                    if fa * fx_new <= 0:
                        b = x_new
                        fb = fx_new
                    else:
                        a = x_new
                        fa = fx_new
                    pts = [(a, fa), (x_new, fx_new), (b, fb)]
                    if abs(b - a) < 1e-15 * max(1.0, abs(x_new)):
                        candidates = [(a, fa), (x_new, fx_new), (b, fb)]
                        best_local = min(candidates, key=lambda p: abs(p[1]))
                        return _polish(f, best_local[0], deadline_exceeded)
                return _polish(f, best_x, deadline_exceeded)
        
        # No bracket found, return best
        return _polish(f, best_x, deadline_exceeded)


def _polish(f, x, deadline_exceeded):
    """High-precision Newton polish to ensure |f(x)| <= 1e-8."""
    if not math.isfinite(x):
        return x
    fx = f(x)
    if not math.isfinite(fx):
        return x
    if abs(fx) <= 1e-8:
        return x
    
    for _ in range(10):
        if deadline_exceeded():
            break
        h = 1e-8 * max(1.0, abs(x))
        d = (f(x + h) - fx) / h
        if d == 0 or not math.isfinite(d):
            break
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
        fx_new = f(x_new)
        if not math.isfinite(fx_new):
            break
        if abs(fx_new) < abs(fx):
            x = x_new
            fx = fx_new
            if abs(fx) <= 1e-8:
                break
        else:
            break
    return x
# EVOLVE-BLOCK-END
