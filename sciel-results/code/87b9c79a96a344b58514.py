import math
import time

_START_TIME = time.time()
_WALL_LIMIT = 115.0  # safety margin before 120s

def _time_left():
    return _WALL_LIMIT - (time.time() - _START_TIME)

def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)

def _lagrange_eval(xs, ys, x):
    """Evaluate Lagrange interpolating polynomial at x."""
    n = len(xs)
    result = 0.0
    for i in range(n):
        term = ys[i]
        for j in range(n):
            if j != i:
                term *= (x - xs[j]) / (xs[i] - xs[j])
        result += term
    return result

def _lagrange_roots_quadratic(x0, y0, x1, y1, x2, y2):
    """Find roots of quadratic interpolating polynomial through 3 points.
    Returns list of real roots."""
    # Convert to standard form: a*x^2 + b*x + c = 0
    # Using Lagrange form, we can compute coefficients
    # P(x) = y0 * L0(x) + y1 * L1(x) + y2 * L2(x)
    # L0(x) = (x-x1)(x-x2) / ((x0-x1)(x0-x2))
    # L1(x) = (x-x0)(x-x2) / ((x1-x0)(x1-x2))
    # L2(x) = (x-x0)(x-x1) / ((x2-x0)(x2-x1))
    
    d00 = (x0 - x1) * (x0 - x2)
    d10 = (x1 - x0) * (x1 - x2)
    d20 = (x2 - x0) * (x2 - x1)
    
    if abs(d00) < 1e-300 or abs(d10) < 1e-300 or abs(d20) < 1e-300:
        return []
    
    # Coefficients of x^2, x, constant
    # L0: (x-x1)(x-x2)/d00 = (x^2 - (x1+x2)x + x1*x2)/d00
    # L1: (x-x0)(x-x2)/d10 = (x^2 - (x0+x2)x + x0*x2)/d10
    # L2: (x-x0)(x-x1)/d20 = (x^2 - (x0+x1)x + x0*x1)/d20
    
    a = y0/d00 + y1/d10 + y2/d20
    b = -y0*(x1+x2)/d00 - y1*(x0+x2)/d10 - y2*(x0+x1)/d20
    c = y0*x1*x2/d00 + y1*x0*x2/d10 + y2*x0*x1/d20
    
    if abs(a) < 1e-300:
        # Linear: bx + c = 0
        if abs(b) < 1e-300:
            return []
        return [-c/b]
    
    disc = b*b - 4*a*c
    if disc < 0:
        return []
    sq_disc = math.sqrt(disc)
    r1 = (-b + sq_disc) / (2*a)
    r2 = (-b - sq_disc) / (2*a)
    roots = [r1, r2]
    # Remove duplicates
    if abs(r1 - r2) < 1e-12 * max(1.0, abs(r1), abs(r2)):
        roots = [r1]
    return roots

def _lagrange_roots_cubic(x0, y0, x1, y1, x2, y2, x3, y3):
    """Find real roots of cubic interpolating polynomial through 4 points.
    Uses numpy for robust root finding."""
    try:
        import numpy as np
    except ImportError:
        return []
    
    # Compute coefficients of cubic polynomial
    # P(x) = a*x^3 + b*x^2 + c*x + d
    # Using Lagrange basis polynomials
    
    d00 = (x0-x1)*(x0-x2)*(x0-x3)
    d10 = (x1-x0)*(x1-x2)*(x1-x3)
    d20 = (x2-x0)*(x2-x1)*(x2-x3)
    d30 = (x3-x0)*(x3-x1)*(x3-x2)
    
    if abs(d00) < 1e-300 or abs(d10) < 1e-300 or abs(d20) < 1e-300 or abs(d30) < 1e-300:
        return []
    
    # Each Li(x) is a quadratic in x
    # L0 = (x-x1)(x-x2)(x-x3)/d00
    # Expand: (x^2 - (x1+x2)x + x1*x2)(x-x3) = x^3 - (x1+x2+x3)x^2 + (x1*x2+x1*x3+x2*x3)x - x1*x2*x3
    
    def cubic_coeffs(xi, xj, xk, denom):
        """Coeffs [a,b,c,d] for (x-xj)(x-xk)(x-xl)/denom where xl is the remaining point"""
        # Actually we need all three other points
        pass
    
    # Let's just build the polynomial directly
    # P(x) = sum_i yi * Li(x)
    # Li(x) = prod_{j!=i} (x - xj) / prod_{j!=i} (xi - xj)
    
    xs = [x0, x1, x2, x3]
    ys = [y0, y1, y2, y3]
    
    # Build coefficient array for P(x) = c3*x^3 + c2*x^2 + c1*x + c0
    coeffs = [0.0, 0.0, 0.0, 0.0]  # c3, c2, c1, c0
    
    for i in range(4):
        denom = 1.0
        # Build numerator polynomial prod_{j!=i} (x - xj)
        num_coeffs = [1.0]  # starts as 1
        for j in range(4):
            if j == i:
                continue
            denom *= (xs[i] - xs[j])
            # Multiply num_coeffs by (x - xs[j])
            # num_coeffs represents a polynomial; multiply by (x - xs[j])
            new_coeffs = [0.0] * (len(num_coeffs) + 1)
            for k in range(len(num_coeffs)):
                new_coeffs[k] += num_coeffs[k] * (-xs[j])
                new_coeffs[k+1] += num_coeffs[k]
            num_coeffs = new_coeffs
        # num_coeffs is [c0, c1, c2, c3] for the numerator (low to high degree)
        if abs(denom) < 1e-300:
            continue
        for k in range(len(num_coeffs)):
            coeffs[k] += ys[i] * num_coeffs[k] / denom
    
    # coeffs = [c0, c1, c2, c3] (low to high degree)
    # For numpy.roots, we need high to low: [c3, c2, c1, c0]
    np_coeffs = [coeffs[3], coeffs[2], coeffs[1], coeffs[0]]
    
    try:
        roots = np.roots(np_coeffs)
    except Exception:
        return []
    
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10 * max(1.0, abs(r.real)):
            real_roots.append(r.real)
    
    return real_roots

def _try_polynomial_interpolation(f, x0):
    """
    Strategy: Sample a few points, fit polynomial, find its roots analytically.
    This can converge in very few function calls if the polynomial degree is low.
    """
    best_x = float(x0)
    best_fx = None
    
    try:
        fx0 = f(x0)
    except Exception:
        return x0
    if not _is_finite(fx0):
        return x0
    best_fx = fx0
    if abs(fx0) <= 1e-8:
        return x0
    
    # Collect points with diverse spacing to capture polynomial shape
    scale = max(1.0, abs(x0))
    
    # Try multiple candidate sets of points
    # Set 1: local neighborhood (quadratic interpolation)
    points = []
    values = []
    
    # Add x0
    points.append(x0)
    values.append(fx0)
    
    # Add neighbors
    candidates = [
        x0 - 0.1 * scale,
        x0 + 0.1 * scale,
        x0 - 1.0 * scale,
        x0 + 1.0 * scale,
        x0 - 0.5 * scale,
        x0 + 0.5 * scale,
        x0 - 2.0 * scale,
        x0 + 2.0 * scale,
    ]
    
    for cx in candidates:
        if len(points) >= 5:
            break
        if not _is_finite(cx):
            continue
        if any(abs(cx - p) < 1e-12 * max(1.0, abs(cx), abs(p)) for p in points):
            continue
        try:
            fcx = f(cx)
        except Exception:
            continue
        if not _is_finite(fcx):
            continue
        points.append(cx)
        values.append(fcx)
        if abs(fcx) < abs(best_fx):
            best_x = cx
            best_fx = fcx
        if abs(fcx) <= 1e-8:
            return cx
    
    n_points = len(points)
    
    # Try quadratic interpolation (3 points)
    if n_points >= 3:
        # Pick 3 points that are well-separated
        # Use first 3 or try different combinations
        for i in range(min(3, n_points)):
            for j in range(i+1, min(3, n_points)):
                for k in range(j+1, min(3, n_points)):
                    if n_points < 3:
                        break
                    p0, p1, p2 = points[i], points[j], points[k]
                    v0, v1, v2 = values[i], values[j], values[k]
                    
                    # Check for sign change
                    if v0 * v2 <= 0 or v0 * v1 <= 0 or v1 * v2 <= 0:
                        roots = _lagrange_roots_quadratic(p0, v0, p1, v1, p2, v2)
                        for r in roots:
                            if _is_finite(r):
                                try:
                                    fr = f(r)
                                except Exception:
                                    continue
                                if _is_finite(fr):
                                    if abs(fr) < abs(best_fx):
                                        best_x = r
                                        best_fx = fr
                                    if abs(fr) <= 1e-8:
                                        return r
    
    # Try cubic interpolation (4 points)
    if n_points >= 4:
        # Try all combinations of 4 points
        import itertools
        for combo in itertools.combinations(range(n_points), 4):
            if _time_left() < 5.0:
                break
            p0, p1, p2, p3 = [points[i] for i in combo]
            v0, v1, v2, v3 = [values[i] for i in combo]
            
            # Check for sign change among these 4
            has_sign_change = False
            for a in range(4):
                for b in range(a+1, 4):
                    if values[combo[a]] * values[combo[b]] <= 0:
                        has_sign_change = True
                        break
                if has_sign_change:
                    break
            
            if not has_sign_change:
                continue
            
            roots = _lagrange_roots_cubic(p0, v0, p1, v1, p2, v2, p3, v3)
            for r in roots:
                if not _is_finite(r):
                    continue
                try:
                    fr = f(r)
                except Exception:
                    continue
                if not _is_finite(fr):
                    continue
                if abs(fr) < abs(best_fx):
                    best_x = r
                    best_fx = fr
                if abs(fr) <= 1e-8:
                    return r
    
    # If polynomial interpolation didn't find a root, try bisection on best bracket
    # Find a bracket from our points
    bracket = None
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if values[i] * values[j] <= 0:
                bracket = (points[i], points[j])
                break
        if bracket:
            break
    
    if bracket:
        a, b = bracket
        fa, fb = None, None
        # We already have these values
        # Bisection
        for _ in range(60):
            if _time_left() < 2.0:
                break
            m = (a + b) / 2
            try:
                fm = f(m)
            except Exception:
                break
            if not _is_finite(fm):
                break
            if abs(fm) < abs(best_fx):
                best_x = m
                best_fx = fm
            if abs(fm) <= 1e-8:
                return m
            if fa is None:
                fa = values[points.index(a)] if a in points else None
            if values[points.index(a)] * fm <= 0 if a in points else None:
                b = m
            else:
                a = m
    else:
        # No bracket found; try secant as fallback
        if len(points) >= 2:
            x_prev = points[0]
            fx_prev = values[0]
            x = points[1]
            fx = values[1]
            for _ in range(30):
                if _time_left() < 2.0:
                    break
                denom = fx - fx_prev
                if abs(denom) < 1e-300:
                    break
                x_new = x - fx * (x - x_prev) / denom
                if not _is_finite(x_new):
                    break
                try:
                    fx_new = f(x_new)
                except Exception:
                    break
                if not _is_finite(fx_new):
                    break
                if abs(fx_new) < abs(best_fx):
                    best_x = x_new
                    best_fx = fx_new
                if abs(fx_new) <= 1e-8:
                    return x_new
                x_prev = x
                fx_prev = fx
                x = x_new
                fx = fx_new
    
    return best_x

def find_root(f, x0):
    x = float(x0)
    
    # Check initial point
    try:
        fx = f(x)
    except Exception:
        return x
    if not _is_finite(fx):
        return x
    if abs(fx) <= 1e-8:
        return x
    
    # Use polynomial interpolation strategy
    result = _try_polynomial_interpolation(f, x0)
    
    if _is_finite(result):
        return result
    
    return x
