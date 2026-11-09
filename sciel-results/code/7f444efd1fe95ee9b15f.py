"""Root finder for smooth polynomial equations with call-budget awareness."""
import time
import math

_DEADLINE_START = time.time()
_WALL_LIMIT = 115.0  # safety margin before 120s kill
_SAFETY_MARGIN = 5.0


def _time_left():
    return _WALL_LIMIT - _SAFETY_MARGIN - (time.time() - _DEADLINE_START)


def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _poly_root_cubic(a, b, c, d):
    """
    Find real roots of ax^3 + bx^2 + cx + d = 0.
    Returns list of real roots.
    """
    if abs(a) < 1e-15:
        # Quadratic or lower
        if abs(b) < 1e-15:
            if abs(c) < 1e-15:
                return []
            return [-d / c]
        # Quadratic: bx^2 + cx + d = 0
        disc = c * c - 4 * b * d
        if disc < -1e-15:
            return []
        elif abs(disc) < 1e-15:
            return [-c / (2 * b)]
        else:
            sq = math.sqrt(disc)
            return [(-c + sq) / (2 * b), (-c - sq) / (2 * b)]
    
    # Normalize
    b = b / a
    c = c / a
    d = d / a
    
    # Depressed cubic: t^3 + pt + q where x = t - b/3
    p = c - b * b / 3.0
    q = d + b * b * b / 27.0 - b * c / 3.0
    
    disc = (q / 2.0) ** 2 + (p / 3.0) ** 3
    
    roots = []
    if disc > 1e-15:
        # One real root
        sq = math.sqrt(disc)
        u = math.copysign(abs(-q / 2.0 + sq) ** (1/3), -q / 2.0 + sq)
        v = math.copysign(abs(-q / 2.0 - sq) ** (1/3), -q / 2.0 - sq)
        t1 = u + v
        roots.append(t1 - b / 3.0)
    elif abs(disc) < 1e-15:
        # Multiple roots
        if abs(q) < 1e-15 and abs(p) < 1e-15:
            roots.append(-b / 3.0)
        else:
            u = math.copysign(abs(-q / 2.0) ** (1/3), -q / 2.0)
            roots.append(2 * u - b / 3.0)
            roots.append(-u - b / 3.0)
    else:
        # Three real roots (trigonometric method)
        r = math.sqrt(-p ** 3 / 27.0)
        phi = math.acos(max(-1.0, min(1.0, -q / (2.0 * r))))
        m = 2.0 * math.sqrt(-p / 3.0)
        for k in range(3):
            t = m * math.cos(phi / 3.0 - 2.0 * math.pi * k / 3.0)
            roots.append(t - b / 3.0)
    
    return roots


def _poly_root_quartic(a, b, c, d, e):
    """
    Find real roots of ax^4 + bx^3 + cx^2 + dx + e = 0.
    Uses numerical approach via deflation.
    """
    if abs(a) < 1e-15:
        return _poly_root_cubic(b, c, d, e)
    
    # Normalize
    b = b / a
    c = c / a
    d = d / a
    e = e / a
    
    # Use companion matrix eigenvalue approach via numpy if available,
    # otherwise use a simple iterative method
    try:
        import numpy as np
        # Companion matrix
        comp = np.array([
            [0, 0, 0, -e],
            [1, 0, 0, -d],
            [0, 1, 0, -c],
            [0, 0, 1, -b]
        ])
        eigenvalues = np.linalg.eigvals(comp)
        real_roots = []
        for ev in eigenvalues:
            if abs(ev.imag) < 1e-10:
                real_roots.append(float(ev.real))
        return real_roots
    except:
        # Fallback: use Newton's method to find roots
        roots = []
        # Try multiple starting points
        for start in [-10, -1, 0, 1, 10]:
            x = float(start)
            for _ in range(50):
                fx = a * x**4 + b * x**3 + c * x**2 + d * x + e
                fpx = 4*a*x**3 + 3*b*x**2 + 2*c*x + d
                if abs(fpx) < 1e-15:
                    break
                x_new = x - fx / fpx
                if abs(x_new - x) < 1e-12:
                    break
                x = x_new
            fx = a * x**4 + b * x**3 + c * x**2 + d * x + e
            if abs(fx) < 1e-8:
                # Check if this root is already found
                is_new = True
                for r in roots:
                    if abs(r - x) < 1e-6:
                        is_new = False
                        break
                if is_new:
                    roots.append(x)
        return roots


def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using polynomial interpolation
    and adaptive bracketing strategies.
    """
    if not _is_finite(x0):
        x0 = 0.0
    x = float(x0)

    # Track best solution found so far
    best_x = x
    best_f = f(x)
    call_count = 1

    def safe_eval(x_val):
        nonlocal call_count, best_x, best_f
        if not _is_finite(x_val):
            return None
        fx_val = f(x_val)
        call_count += 1
        if not _is_finite(fx_val):
            return None
        if abs(fx_val) < abs(best_f):
            best_f = fx_val
            best_x = x_val
        return fx_val

    # Initial evaluation
    fx = best_f
    if abs(fx) <= 1e-8:
        return best_x

    # Strategy: Collect points and use polynomial interpolation
    points = [(x, fx)]
    
    # Try to find a bracket first
    h0 = 1e-4 * max(1.0, abs(x))
    x2 = x + h0
    fx2 = safe_eval(x2)
    
    if fx2 is None:
        x2 = x - h0
        fx2 = safe_eval(x2)
        
    if fx2 is None:
        return best_x
    
    points.append((x2, fx2))
    
    if fx * fx2 < 0:
        # We have a bracket - use Illinois algorithm
        a, b = x, x2
        fa, fb = fx, fx2
        for _ in range(60):
            if call_count >= 480 or _time_left() <= 0:
                break
            if abs(fb - fa) < 1e-15:
                m = (a + b) / 2.0
            else:
                m = b - fb * (b - a) / (fb - fa)
            
            if not (a < m < b):
                m = (a + b) / 2.0
                
            fm = safe_eval(m)
            if fm is None:
                break
                
            if abs(fm) <= 1e-8:
                return m

            if fa * fm < 0:
                b, fb = m, fm
                fa = fa * 0.5
            else:
                a, fa = m, fm
                fb = fb * 0.5
        
        if abs(best_f) <= 1e-8:
            return best_x
        return best_x
    
    # No sign change - use polynomial interpolation
    # Collect more points to fit a polynomial
    for i in range(3, 6):
        if call_count >= 480 or _time_left() <= 0:
            break
        # Try different step sizes
        step = h0 * (10 ** (i - 3))
        xi = x + step
        fxi = safe_eval(xi)
        if fxi is not None:
            points.append((xi, fxi))
        
        xi = x - step
        fxi = safe_eval(xi)
        if fxi is not None:
            points.append((xi, fxi))
    
    # Remove duplicates
    unique_points = []
    for px, fpx in points:
        is_dup = False
        for ux, fux in unique_points:
            if abs(px - ux) < 1e-12:
                is_dup = True
                break
        if not is_dup:
            unique_points.append((px, fpx))
    
    points = unique_points
    
    # Try to fit polynomials of increasing degree
    for deg in [3, 4]:
        if len(points) < deg + 1:
            continue
        
        # Select deg+1 points closest to x0
        sorted_pts = sorted(points, key=lambda p: abs(p[0] - x0))
        selected = sorted_pts[:deg + 1]
        
        # Build Vandermonde matrix and solve for coefficients
        # Use numpy if available
        try:
            import numpy as np
            A = np.vander([p[0] for p in selected], deg + 1, increasing=True)
            y = np.array([p[1] for p in selected])
            
            # Solve A @ coeffs = y
            coeffs, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
            
            # Find roots of the polynomial
            if deg == 3:
                # coeffs = [a0, a1, a2, a3] for a0 + a1*x + a2*x^2 + a3*x^3
                roots = _poly_root_cubic(coeffs[3], coeffs[2], coeffs[1], coeffs[0])
            else:
                roots = _poly_root_quartic(coeffs[4], coeffs[3], coeffs[2], coeffs[1], coeffs[0])
            
            # Evaluate candidate roots
            for r in roots:
                if not _is_finite(r):
                    continue
                if call_count >= 480 or _time_left() <= 0:
                    break
                fr = safe_eval(r)
                if fr is None:
                    continue
                if abs(fr) <= 1e-8:
                    return r
        except:
            pass
    
    # If polynomial interpolation didn't work, fall back to secant method
    if len(points) >= 2:
        # Use two closest points for secant
        sorted_pts = sorted(points, key=lambda p: abs(p[0] - x0))
        x1, f1 = sorted_pts[0]
        x2, f2 = sorted_pts[1]
        
        for _ in range(40):
            if call_count >= 480 or _time_left() <= 0:
                break
            if abs(f2 - f1) < 1e-15:
                break
            
            x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
            if not _is_finite(x_new):
                break
            
            f_new = safe_eval(x_new)
            if f_new is None:
                break
                
            if abs(f_new) <= 1e-8:
                return x_new
                
            if abs(f_new - f2) < 1e-15 and abs(x_new - x2) < 1e-12:
                break
                
            x1, f1 = x2, f2
            x2, f2 = x_new, f_new
    
    return best_x
