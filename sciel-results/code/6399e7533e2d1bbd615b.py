import math
import time

def find_root(f, x0):
    start_time = time.monotonic()
    deadline = start_time + 100.0  # Safety margin before 150s kill
    call_count = 0
    max_calls = 480  # Keep under 500 to avoid penalty

    def safe_f(x):
        nonlocal call_count
        if call_count >= max_calls:
            return None
        if not math.isfinite(x):
            return None
        call_count += 1
        try:
            val = f(x)
            if not math.isfinite(val):
                return None
            return val
        except Exception:
            return None

    x = float(x0)
    fx = safe_f(x)
    if fx is None:
        return x
    if abs(fx) <= 1e-8:
        return x

    # Try to find a bracket [a, b] with sign change
    a = x - 1.0
    b = x + 1.0
    fa = safe_f(a)
    fb = safe_f(b)
    
    if fa is None or fb is None:
        # Fallback to local search without bracketing
        for _ in range(50):
            if time.monotonic() > deadline:
                break
            h = 1e-5 * max(1.0, abs(x))
            fx2 = safe_f(x + h)
            if fx2 is None:
                break
            d = (fx2 - fx) / h
            if abs(d) < 1e-15:
                break
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            if abs(x_new - x) > 10 * max(1.0, abs(x)):
                x_new = x - math.copysign(0.1 * max(1.0, abs(x)), fx)
            x = x_new
            fx = safe_f(x)
            if fx is None:
                break
            if abs(fx) <= 1e-8:
                return x
        return x

    # Expand bracket if needed
    range_exp = 1.0
    for _ in range(25):
        if fa is not None and fb is not None and fa * fb < 0:
            break
        range_exp *= 2.0
        a = x - range_exp
        b = x + range_exp
        fa = safe_f(a)
        fb = safe_f(b)
        if fa is None or fb is None:
            break
        if time.monotonic() > deadline:
            break
    else:
        # If no bracket found, use local damped Newton
        for _ in range(50):
            if time.monotonic() > deadline:
                break
            h = 1e-5 * max(1.0, abs(x))
            fx2 = safe_f(x + h)
            if fx2 is None:
                break
            d = (fx2 - fx) / h
            if abs(d) < 1e-15:
                break
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            if abs(x_new - x) > 10 * max(1.0, abs(x)):
                x_new = x - math.copysign(0.1 * max(1.0, abs(x)), fx)
            x = x_new
            fx = safe_f(x)
            if fx is None:
                break
            if abs(fx) <= 1e-8:
                return x
        return x

    # Ensure we have a valid bracket
    if fa is None or fb is None or fa * fb >= 0:
        # Fallback: use best point
        return x

    # Use Inverse Quadratic Interpolation (IQI) within the bracket
    # We maintain three points and fit a quadratic to (x, f(x))
    # Then find where the quadratic crosses zero.
    
    # Initialize with a, x, b
    # We need distinct points. Let's use a, x, b if they are distinct enough.
    # If x is close to a or b, we might need to adjust.
    # For simplicity, let's use the bisection point if x is too close to an endpoint.
    
    points = []
    fvals = []
    
    def add_point(px, fpx):
        if fpx is None:
            return
        # Check if point is too close to existing points
        for i, (qx, qf) in enumerate(zip(points, fvals)):
            if abs(px - qx) < 1e-10 * max(1.0, abs(px), abs(qx)):
                # If new point is better (smaller |f|), replace
                if abs(fpx) < abs(qf):
                    points[i] = px
                    fvals[i] = fpx
                return
        points.append(px)
        fvals.append(fpx)
        # Keep only the last 3 points for IQI, but ensure bracket is maintained
        # Actually, for IQI we need 3 points. Let's keep the 3 most recent distinct ones.
        # But we also need to maintain the bracket [a, b].
        # Let's maintain a separate bracket [a_br, b_br] and a list of recent points.
    
    # Let's restart with a cleaner structure:
    # Maintain bracket [a_br, b_br] with f(a_br), f(b_br).
    # Maintain a list of up to 3 recent points for IQI.
    
    a_br = a
    b_br = b
    fa_br = fa
    fb_br = fb
    
    # Ensure a_br < b_br
    if a_br > b_br:
        a_br, b_br = b_br, a_br
        fa_br, fb_br = fb_br, fa_br
    
    # Recent points for IQI: start with a_br, x, b_br
    recent_x = [a_br, x, b_br]
    recent_f = [fa_br, fx, fb_br]
    
    # Check if any recent_f is None
    for i in range(len(recent_f)):
        if recent_f[i] is None:
            recent_f[i] = 0.0  # Should not happen if safe_f worked
    
    # Check if points are distinct enough
    # If not, we might need to generate new points via bisection.
    
    for _ in range(100):
        if time.monotonic() > deadline:
            break
        
        # Check convergence on bracket
        if abs(b_br - a_br) < 1e-10 * max(1.0, abs(a_br), abs(b_br)):
            return (a_br + b_br) / 2.0
        
        # Check if any recent point is a root
        for i in range(len(recent_x)):
            if abs(recent_f[i]) <= 1e-8:
                return recent_x[i]
        
        # Try IQI
        # Fit quadratic through (x0, f0), (x1, f1), (x2, f2)
        # We want x such that q(x) = 0.
        # q(x) = A x^2 + B x + C
        # We can use Lagrange interpolation or solve the linear system.
        # Alternatively, use the formula for inverse quadratic interpolation.
        # Let's use the formula:
        # x_new = x2 + (x2-x1)(x2-x0) * f2 / ( (x2-x1)(f2-f1) - (x2-x0)(f2-f0) ) ... wait, that's for secant-like.
        # Actually, IQI fits q(x) to the 3 points and finds the root.
        # Let's use numpy.polyfit or manual calculation.
        # Manual calculation to avoid numpy overhead:
        # Solve for A, B, C:
        # A x0^2 + B x0 + C = f0
        # A x1^2 + B x1 + C = f1
        # A x2^2 + B x2 + C = f2
        # This is a 3x3 system.
        
        x0, x1, x2 = recent_x[0], recent_x[1], recent_x[2]
        f0, f1, f2 = recent_f[0], recent_f[1], recent_f[2]
        
        # Check if points are distinct
        if abs(x1 - x0) < 1e-12 or abs(x2 - x1) < 1e-12 or abs(x2 - x0) < 1e-12:
            # Points are too close, bisect
            x_mid = (a_br + b_br) / 2.0
            f_mid = safe_f(x_mid)
            if f_mid is None:
                break
            if fa_br * f_mid <= 0:
                b_br = x_mid
                fb_br = f_mid
            else:
                a_br = x_mid
                fa_br = f_mid
            # Update recent points: add x_mid, keep last 3
            recent_x.append(x_mid)
            recent_f.append(f_mid)
            if len(recent_x) > 3:
                recent_x.pop(0)
                recent_f.pop(0)
            continue
        
        # Solve for A, B, C
        # Using Cramer's rule or Gaussian elimination.
        # Let's use a simple formula for the root of the interpolating quadratic.
        # The root is given by:
        # x_new = x2 - (x2-x1)(x2-x0) * f2 / ( (x2-x1)(f2-f1) - (x2-x0)(f2-f0) ) ... no, that's not right.
        
        # Let's use the standard IQI formula.
        # h01 = f1 - f0
        # h12 = f2 - f1
        # h02 = f2 - f0
        # x01 = x1 - x0
        # x12 = x2 - x1
        # x02 = x2 - x0
        # x_new = x2 + (x2-x1)(x2-x0) * f2 / ( (x2-x1)(f2-f1) - (x2-x0)(f2-f0) ) ... wait.
        
        # Actually, let's just solve the 3x3 system.
        # [x0^2, x0, 1] [A]   [f0]
        # [x1^2, x1, 1] [B] = [f1]
        # [x2^2, x2, 1] [C]   [f2]
        
        # Determinant:
        det = (x0*x1*x2 - x0*x1 - x0*x2 + x0 - x1*x2 + x1 + x2) * 0 # This is getting complicated.
        
        # Let's use numpy if available, otherwise fallback to secant/bisection.
        try:
            import numpy as np
            X = np.array([[x0*x0, x0, 1], [x1*x1, x1, 1], [x2*x2, x2, 1]])
            Y = np.array([f0, f1, f2])
            coeffs = np.linalg.solve(X, Y)
            A, B, C = coeffs
            # Solve A x^2 + B x + C = 0
            disc = B*B - 4*A*C
            if disc < 0:
                # No real root, bisect
                x_mid = (a_br + b_br) / 2.0
                f_mid = safe_f(x_mid)
                if f_mid is None:
                    break
                if fa_br * f_mid <= 0:
                    b_br = x_mid
                    fb_br = f_mid
                else:
                    a_br = x_mid
                    fa_br = f_mid
                recent_x.append(x_mid)
                recent_f.append(f_mid)
                if len(recent_x) > 3:
                    recent_x.pop(0)
                    recent_f.pop(0)
                continue
            sqrt_disc = math.sqrt(disc)
            x_sol1 = (-B + sqrt_disc) / (2*A)
            x_sol2 = (-B - sqrt_disc) / (2*A)
            # Choose the solution in the bracket
            if a_br < x_sol1 < b_br:
                x_new = x_sol1
            elif a_br < x_sol2 < b_br:
                x_new = x_sol2
            else:
                # Bisect
                x_new = (a_br + b_br) / 2.0
        except Exception:
            # Fallback to secant
            if abs(f1 - f0) > 1e-15:
                x_new = x1 - f1 * (x1 - x0) / (f1 - f0)
            else:
                x_new = (a_br + b_br) / 2.0
        
        if not math.isfinite(x_new):
            x_new = (a_br + b_br) / 2.0
        
        # Check if x_new is in bracket
        if not (a_br < x_new < b_br):
            x_new = (a_br + b_br) / 2.0
        
        f_new = safe_f(x_new)
        if f_new is None:
            break
        
        if abs(f_new) <= 1e-8:
            return x_new
        
        # Update bracket
        if fa_br * f_new <= 0:
            b_br = x_new
            fb_br = f_new
        else:
            a_br = x_new
            fa_br = f_new
        
        # Update recent points
        recent_x.append(x_new)
        recent_f.append(f_new)
        if len(recent_x) > 3:
            recent_x.pop(0)
            recent_f.pop(0)
        
        # Check convergence
        if abs(b_br - a_br) < 1e-10 * max(1.0, abs(a_br), abs(b_br)):
            return (a_br + b_br) / 2.0

    # Return best point
    best_x = x
    best_f = abs(fx)
    for i in range(len(recent_x)):
        if abs(recent_f[i]) < best_f:
            best_f = abs(recent_f[i])
            best_x = recent_x[i]
    return best_x
