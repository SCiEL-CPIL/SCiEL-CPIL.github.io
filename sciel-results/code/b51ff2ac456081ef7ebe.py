import math
import time

def find_root(f, x0):
    """
    Find root x of f(x) such that |f(x)| <= 1e-8.
    Uses cubic Lagrange interpolation to predict root, then safeguarded Newton refinement.
    """
    start_time = time.time()
    deadline = start_time + 10.0
    tol_final = 1e-8

    x0 = float(x0)
    f0 = f(x0)
    if math.isfinite(f0) and abs(f0) <= tol_final:
        return x0

    # Handle non-finite f0
    if not math.isfinite(f0):
        base_step = 1e-4 * max(1.0, abs(x0))
        for s in [1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32, 64, -64]:
            if time.time() > deadline:
                break
            x_try = x0 + s * base_step
            f_try = f(x_try)
            if math.isfinite(f_try) and abs(f_try) <= tol_final:
                return x_try
            if math.isfinite(f_try):
                x0, f0 = x_try, f_try
                break
        if not math.isfinite(f0):
            return x0

    # Build a set of points for interpolation
    # Start with x0 and expand outward to find sign changes
    base_step = 1e-3 * max(1.0, abs(x0))
    points = [(x0, f0)]
    
    # Try to find a sign change by expanding
    left_x, left_f = x0, f0
    right_x, right_f = x0, f0
    found_left = False
    found_right = False
    
    for i in range(1, 20):
        if time.time() > deadline - 5.0:
            break
        step = base_step * (2 ** i)
        
        # Try right
        if not found_right:
            xr = x0 + step
            fr = f(xr)
            if math.isfinite(fr):
                right_x, right_f = xr, fr
                points.append((xr, fr))
                if abs(fr) <= tol_final:
                    return xr
                if f0 * fr < 0:
                    found_right = True
        
        # Try left
        if not found_left:
            xl = x0 - step
            fl = f(xl)
            if math.isfinite(fl):
                left_x, left_f = xl, fl
                points.append((xl, fl))
                if abs(fl) <= tol_final:
                    return xl
                if f0 * fl < 0:
                    found_left = True
        
        if found_left and found_right:
            break
    
    # If we have a bracket, use bisection + Newton hybrid
    if found_left and found_right:
        # We have a bracket [left_x, right_x]
        a, fa = left_x, left_f
        b, fb = right_x, right_f
        
        # Ensure fa * fb < 0
        if fa * fb > 0:
            # Swap if needed
            a, b = b, a
            fa, fb = fb, fa
        
        # Safeguarded Newton/bisection
        x = a + (b - a) * 0.5
        fx = f(x)
        
        for _ in range(50):
            if time.time() > deadline - 1.0:
                break
            if abs(fx) <= tol_final:
                return x
            
            # Try Newton step
            # Estimate derivative using finite difference
            h = 1e-8 * max(1.0, abs(x))
            fp = (f(x + h) - f(x - h)) / (2 * h)
            if math.isfinite(fp) and abs(fp) > 1e-15:
                x_newton = x - fx / fp
                if a < x_newton < b:
                    x = x_newton
                    fx = f(x)
                    continue
            
            # Bisection step
            if fa * fx < 0:
                b, fb = x, fx
            else:
                a, fa = x, fx
            x = a + (b - a) * 0.5
            fx = f(x)
        
        return x
    
    # No bracket found, use cubic interpolation on available points
    # Filter finite points
    finite_points = [(x, fx) for x, fx in points if math.isfinite(fx)]
    if len(finite_points) < 4:
        # Pad with more points if needed
        for i in range(1, 10):
            if time.time() > deadline - 2.0:
                break
            step = base_step * (2 ** (i + 20))
            for sign in [1, -1]:
                xt = x0 + sign * step
                ft = f(xt)
                if math.isfinite(ft):
                    finite_points.append((xt, ft))
                    if len(finite_points) >= 4:
                        break
            if len(finite_points) >= 4:
                break
    
    if len(finite_points) < 4:
        # Fallback: return best point
        best_x = x0
        best_f = abs(f0)
        for xt, ft in finite_points:
            if abs(ft) < best_f:
                best_x = xt
                best_f = abs(ft)
        return best_x
    
    # Use first 4 points for cubic Lagrange interpolation
    pts = finite_points[:4]
    x1, f1 = pts[0]
    x2, f2 = pts[1]
    x3, f3 = pts[2]
    x4, f4 = pts[3]
    
    # Cubic Lagrange polynomial P(x) = sum fi * Li(x)
    # Find root of P(x) = 0
    # P(x) = a*x^3 + b*x^2 + c*x + d
    # Solve using numpy or manual cubic formula
    
    # Use numpy for cubic root finding
    try:
        import numpy as np
        
        # Build Lagrange basis polynomials
        # L1(x) = (x-x2)(x-x3)(x-x4) / ((x1-x2)(x1-x3)(x1-x4))
        # etc.
        
        # Coefficients of each Li
        def lagrange_coeff(xi, xj, xk, xl):
            # (x-xj)(x-xk)(x-xl) = x^3 - (xj+xk+xl)x^2 + (xj*xk+xj*xl+xk*xl)x - xj*xk*xl
            a = 1.0
            b = -(xj + xk + xl)
            c = xj*xk + xj*xl + xk*xl
            d = -xj*xk*xl
            denom = (xi - xj) * (xi - xk) * (xi - xl)
            if abs(denom) < 1e-15:
                return [0, 0, 0, 0]
            return [a/denom, b/denom, c/denom, d/denom]
        
        c1 = lagrange_coeff(x1, x2, x3, x4)
        c2 = lagrange_coeff(x2, x1, x3, x4)
        c3 = lagrange_coeff(x3, x1, x2, x4)
        c4 = lagrange_coeff(x4, x1, x2, x3)
        
        # P(x) = f1*c1 + f2*c2 + f3*c3 + f4*c4
        a = f1*c1[0] + f2*c2[0] + f3*c3[0] + f4*c4[0]
        b = f1*c1[1] + f2*c2[1] + f3*c3[1] + f4*c4[1]
        c = f1*c1[2] + f2*c2[2] + f3*c3[2] + f4*c4[2]
        d = f1*c1[3] + f2*c2[3] + f3*c3[3] + f4*c4[3]
        
        # Find real roots of a*x^3 + b*x^2 + c*x + d = 0
        if abs(a) < 1e-15:
            # Quadratic
            if abs(b) < 1e-15:
                # Linear
                if abs(c) < 1e-15:
                    return x0
                roots = [-d/c]
            else:
                disc = c*c - 4*b*d
                if disc < 0:
                    return x0
                roots = [(-c + math.sqrt(disc))/(2*b), (-c - math.sqrt(disc))/(2*b)]
        else:
            # Cubic: use numpy
            coeffs = [a, b, c, d]
            roots = np.roots(coeffs)
            roots = [r.real for r in roots if abs(r.imag) < 1e-10]
        
        # Pick the root closest to x0
        if not roots:
            return x0
        x_pred = min(roots, key=lambda r: abs(r - x0))
        
        if not math.isfinite(x_pred):
            return x0
        
        # Evaluate f at predicted root
        f_pred = f(x_pred)
        if math.isfinite(f_pred) and abs(f_pred) <= tol_final:
            return x_pred
        
        # Use x_pred as starting point for Newton refinement
        x = x_pred
        fx = f_pred
        
        # Need a second point for secant/Newton
        h = 1e-6 * max(1.0, abs(x))
        x2n = x + h
        f2n = f(x2n)
        
        if not math.isfinite(f2n):
            x2n = x - h
            f2n = f(x2n)
        
        # Secant iterations
        xa, fa = x2n, f2n
        xb, fb = x, fx
        
        for _ in range(30):
            if time.time() > deadline - 0.5:
                break
            if abs(fb) <= tol_final:
                return xb
            if abs(fa) <= tol_final:
                return xa
            
            denom = fb - fa
            if abs(denom) < 1e-15:
                break
            
            xn = xb - fb * (xb - xa) / denom
            if not math.isfinite(xn):
                break
            
            fn = f(xn)
            if not math.isfinite(fn):
                break
            
            if abs(fn) <= tol_final:
                return xn
            
            xa, fa = xb, fb
            xb, fb = xn, fn
            
            if abs(xn - xb) < 1e-14:
                break
        
        # Return best of xa, xb
        if abs(fa) < abs(fb):
            return xa
        return xb
    
    except Exception:
        # Fallback: return best point
        best_x = x0
        best_f = abs(f0)
        for xt, ft in finite_points:
            if abs(ft) < best_f:
                best_x = xt
                best_f = abs(ft)
        return best_x
