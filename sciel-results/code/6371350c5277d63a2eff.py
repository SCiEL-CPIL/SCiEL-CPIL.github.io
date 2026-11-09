"""Root finder using adaptive interpolation with Newton polish."""
import time
import math

_DEADLINE_START = time.time()
_WALL_LIMIT = 115.0
_SAFETY_MARGIN = 5.0


def _time_left():
    return _WALL_LIMIT - _SAFETY_MARGIN - (time.time() - _DEADLINE_START)


def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _quadratic_root(x0, f0, x1, f1, x2, f2):
    """Find root of quadratic through 3 points. Returns best finite root or None."""
    # Divided differences
    if abs(x1 - x0) < 1e-15:
        return None
    if abs(x2 - x0) < 1e-15:
        return None
    if abs(x2 - x1) < 1e-15:
        return None
    
    d01 = (f1 - f0) / (x1 - x0)
    d02 = (f2 - f0) / (x2 - x0)
    d12 = (f2 - f1) / (x2 - x1)
    d012 = (d12 - d01) / (x2 - x0)
    
    a = d012
    b = d01 - a * x0
    c = f0 - b * x0 - a * x0 * x0
    
    if abs(a) < 1e-15:
        if abs(b) < 1e-15:
            return None
        return -c / b
    
    disc = b * b - 4 * a * c
    if disc < -1e-15:
        return None
    disc = max(0.0, disc)
    sq = math.sqrt(disc)
    r1 = (-b + sq) / (2 * a)
    r2 = (-b - sq) / (2 * a)
    
    best = min([r1, r2], key=lambda r: abs(r - x1))
    return best if _is_finite(best) else None


def _cubic_root(x0, f0, x1, f1, x2, f2, x3, f3):
    """Find root of cubic polynomial interpolating 4 points."""
    if abs(x1 - x0) < 1e-15 or abs(x2 - x0) < 1e-15 or abs(x3 - x0) < 1e-15 or \
       abs(x2 - x1) < 1e-15 or abs(x3 - x1) < 1e-15 or abs(x3 - x2) < 1e-15:
        return None
    
    d01 = (f1 - f0) / (x1 - x0)
    d12 = (f2 - f1) / (x2 - x1)
    d23 = (f3 - f2) / (x3 - x2)
    d02 = (f2 - f0) / (x2 - x0)
    d13 = (f3 - f1) / (x3 - x1)
    d03 = (f3 - f0) / (x3 - x0)
    
    d012 = (d12 - d01) / (x2 - x0)
    d123 = (d23 - d12) / (x3 - x1)
    d023 = (d23 - d02) / (x3 - x0)
    d0123 = (d123 - d012) / (x3 - x0)
    
    a = d0123
    b = d012 - a * x0
    c = d01 - b * x0 - a * x0 * x0
    d = f0 - c * x0 - b * x0 * x0 - a * x0 * x0 * x0
    
    if abs(a) < 1e-15:
        return _quadratic_root(x0, f0, x1, f1, x2, f2)
    
    # Normalize: x^3 + p*x^2 + q*x + r = 0
    p = b / a
    q = c / a
    r = d / a
    
    # Depressed cubic: x = t - p/3
    p_prime = q - p * p / 3.0
    q_prime = 2.0 * p**3 / 27.0 - p * q / 3.0 + r
    
    disc = (q_prime / 2.0)**2 + (p_prime / 3.0)**3
    
    roots = []
    if disc > 1e-15:
        sq_disc = math.sqrt(disc)
        u = -q_prime / 2.0 + sq_disc
        v = -q_prime / 2.0 - sq_disc
        
        def cbrt(x):
            if x >= 0:
                return x**(1.0/3.0)
            else:
                return -((-x)**(1.0/3.0))
        
        t = cbrt(u) + cbrt(v)
        roots.append(t - p / 3.0)
    elif abs(disc) <= 1e-15:
        if abs(p_prime) < 1e-15:
            roots.append(-p / 3.0)
        else:
            t1 = 3.0 * q_prime / p_prime
            t2 = -3.0 * q_prime / (2.0 * p_prime)
            roots.append(t1 - p / 3.0)
            roots.append(t2 - p / 3.0)
    else:
        rho = math.sqrt(-p_prime**3 / 27.0)
        theta = math.acos(max(-1.0, min(1.0, -q_prime / (2.0 * rho))))
        mag = 2.0 * (-p_prime / 3.0)**0.5
        for k in range(3):
            t = mag * math.cos(theta / 3.0 - 2.0 * math.pi * k / 3.0)
            roots.append(t - p / 3.0)
    
    if not roots:
        return None
    
    best = min(roots, key=lambda r: abs(r - x1))
    return best if _is_finite(best) else None


def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using adaptive interpolation with Newton polish.
    """
    tol = 1e-8
    polish_tol = 1e-12
    max_iter = 60
    
    x0_val = float(x0)
    try:
        f0 = f(x0_val)
    except Exception:
        return x0_val
    
    if not _is_finite(f0):
        return x0_val
    
    best_x = x0_val
    best_f = abs(f0)
    
    if best_f <= tol:
        return x0_val
    
    # Generate initial points with adaptive spacing
    scale = max(1.0, abs(x0_val))
    # Start with small steps, expand if needed
    base_h = 1e-3 * scale
    
    points = [(x0_val, f0)]
    
    # Try positive direction first
    for i, h in enumerate([base_h * 0.1, base_h, base_h * 10.0]):
        if _time_left() <= 0:
            break
        x_try = x0_val + h
        try:
            f_try = f(x_try)
        except Exception:
            break
        if not _is_finite(f_try):
            continue
        abs_f_try = abs(f_try)
        if abs_f_try < best_f:
            best_x = x_try
            best_f = abs_f_try
        points.append((x_try, f_try))
        if best_f <= tol:
            return x_try
    
    # Try negative direction if we don't have enough points
    for i, h in enumerate([-base_h * 0.1, -base_h, -base_h * 10.0]):
        if len(points) >= 4:
            break
        if _time_left() <= 0:
            break
        x_try = x0_val + h
        try:
            f_try = f(x_try)
        except Exception:
            break
        if not _is_finite(f_try):
            continue
        abs_f_try = abs(f_try)
        if abs_f_try < best_f:
            best_x = x_try
            best_f = abs_f_try
        points.append((x_try, f_try))
        if best_f <= tol:
            return x_try
    
    # Keep at most 4 best points (by |f|)
    if len(points) > 4:
        points.sort(key=lambda p: abs(p[1]))
        points = points[:4]
    
    # Main iteration loop
    for _ in range(max_iter):
        if _time_left() <= 0:
            break
        
        # Choose interpolation method based on available points
        if len(points) >= 4:
            (xa, fa), (xb, fb), (xc, fc), (xd, fd) = points
            root = _cubic_root(xa, fa, xb, fb, xc, fc, xd, fd)
        elif len(points) >= 3:
            (xa, fa), (xb, fb), (xc, fc) = points
            root = _quadratic_root(xa, fa, xb, fb, xc, fc)
        elif len(points) >= 2:
            (xa, fa), (xb, fb) = points
            dx = xb - xa
            if dx == 0:
                break
            m = (fb - fa) / dx
            if m == 0 or not _is_finite(m):
                break
            root = xb - fb / m
        else:
            break
        
        if root is None or not _is_finite(root):
            # Fallback: use secant with last two points
            if len(points) >= 2:
                (xa, fa), (xb, fb) = points[-2], points[-1]
                dx = xb - xa
                if dx == 0:
                    break
                m = (fb - fa) / dx
                if m == 0 or not _is_finite(m):
                    break
                root = xb - fb / m
            else:
                break
        
        if root is None or not _is_finite(root):
            break
        
        try:
            f_root = f(root)
        except Exception:
            break
        
        if not _is_finite(f_root):
            continue
        
        abs_f_root = abs(f_root)
        if abs_f_root < best_f:
            best_x = root
            best_f = abs_f_root
        
        if abs_f_root <= tol:
            return root
        
        # Add new point and keep best 4
        points.append((root, f_root))
        if len(points) > 4:
            points.sort(key=lambda p: abs(p[1]))
            points = points[:4]
        
        # Check for convergence stall
        if len(points) > 2 and abs(points[-1][0] - points[-2][0]) < 1e-14:
            # Try Newton polish if we're close
            if best_f < 1e-6:
                return _newton_polish(f, best_x, best_f, polish_tol)
            break
    
    # Final Newton polish if we're reasonably close
    if best_f < 1e-6:
        result = _newton_polish(f, best_x, best_f, polish_tol)
        if result is not None:
            return result
    
    return best_x


def _newton_polish(f, x_start, f_start, tol, max_iter=10):
    """Newton's method with numerical derivative for final polish."""
    if not _is_finite(x_start) or not _is_finite(f_start):
        return None
    
    x = x_start
    fx = f_start
    
    h = 1e-8 * max(1.0, abs(x))
    
    for _ in range(max_iter):
        if _time_left() <= 0:
            break
        
        if abs(fx) <= tol:
            return x
        
        try:
            fx_plus = f(x + h)
        except Exception:
            break
        
        if not _is_finite(fx_plus):
            break
        
        df = (fx_plus - fx) / h
        if not _is_finite(df) or abs(df) < 1e-15:
            break
        
        x_new = x - fx / df
        if not _is_finite(x_new):
            break
        
        try:
            fx_new = f(x_new)
        except Exception:
            break
        
        if not _is_finite(fx_new):
            break
        
        if abs(fx_new) < abs(fx):
            x = x_new
            fx = fx_new
        else:
            # Damped step
            x_new = x - 0.5 * fx / df
            if not _is_finite(x_new):
                break
            try:
                fx_new = f(x_new)
            except Exception:
                break
            if not _is_finite(fx_new):
                break
            if abs(fx_new) < abs(fx):
                x = x_new
                fx = fx_new
        
        if abs(fx) <= tol:
            return x
    
    return x if abs(fx) <= tol else None
