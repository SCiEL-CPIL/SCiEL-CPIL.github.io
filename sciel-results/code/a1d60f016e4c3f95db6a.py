"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_T0 = time.time()
_LIMIT = 110.0
_CALL_LIMIT = 490  # stay well under 500

def _left():
    return _LIMIT - (time.time() - _T0)

def _fin(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)

def _poly_interp_root(xs, fs, degree=None):
    """Fit a polynomial to (xs, fs) and find its real roots closest to the data."""
    n = len(xs)
    if n < 2:
        return None
    if degree is None:
        degree = min(n - 1, 4)
    # Use numpy polyfit
    try:
        coeffs = np.polyfit(xs, fs, degree)
        roots = np.roots(coeffs)
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-10]
        if not real_roots:
            return None
        # Return the real root closest to the mean of xs
        cx = sum(xs) / len(xs)
        real_roots.sort(key=lambda r: abs(r - cx))
        return real_roots[0]
    except Exception:
        return None

def find_root(f, x0):
    x0 = float(x0)
    best_x = x0
    best_fx = float('inf')
    call_count = 0
    
    def _eval(x):
        nonlocal call_count, best_x, best_fx
        call_count += 1
        try:
            fx = f(x)
        except Exception:
            return None
        if not _fin(fx):
            return None
        if abs(fx) < abs(best_fx):
            best_fx, best_x = fx, x
        if abs(fx) <= 1e-8:
            return fx
        return fx
    
    # Evaluate starting point
    fx0 = _eval(x0)
    if fx0 is not None and abs(fx0) <= 1e-8:
        return x0
    
    # Strategy 1: Polynomial interpolation root finding
    # Collect points and use interpolation to predict roots
    if call_count < _CALL_LIMIT and _left() > 2.0:
        # Start with x0 and nearby points
        points = [(x0, fx0)] if fx0 is not None else []
        
        # Add a few more points to build interpolation
        base_scale = max(1.0, abs(x0))
        offsets = [0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0]
        
        for off in offsets:
            if call_count >= _CALL_LIMIT or _left() < 2.0:
                break
            x_try = x0 + off * base_scale
            if not _fin(x_try):
                continue
            fx_try = _eval(x_try)
            if fx_try is not None:
                if abs(fx_try) <= 1e-8:
                    return x_try
                points.append((x_try, fx_try))
        
        # Try polynomial interpolation with increasing degree
        for deg in [2, 3, 4]:
            if call_count >= _CALL_LIMIT or _left() < 2.0:
                break
            if len(points) < deg + 1:
                continue
            
            # Use the most recent/best points
            # Sort by |f(x)| to prioritize points near zero
            sorted_pts = sorted(points, key=lambda p: abs(p[1]))
            # Take deg+1 points closest to zero
            selected = sorted_pts[:deg + 1]
            xs = [p[0] for p in selected]
            fs = [p[1] for p in selected]
            
            root_est = _poly_interp_root(xs, fs, degree=deg)
            if root_est is not None and _fin(root_est):
                fx_root = _eval(root_est)
                if fx_root is not None:
                    if abs(fx_root) <= 1e-8:
                        return root_est
                    # If good, refine with Newton
                    if abs(fx_root) < 1e-4:
                        # Newton refinement
                        x = root_est
                        for i in range(15):
                            if call_count >= _CALL_LIMIT or _left() < 1.0:
                                break
                            fx = _eval(x)
                            if fx is None:
                                break
                            if abs(fx) <= 1e-8:
                                return x
                            h = 1e-6 * max(1.0, abs(x))
                            fxh = _eval(x + h)
                            if fxh is None:
                                break
                            d = (fxh - fx) / h
                            if not _fin(d) or abs(d) < 1e-15:
                                break
                            step = fx / d
                            if abs(step) > 5.0 * max(1.0, abs(x)):
                                step = math.copysign(5.0 * max(1.0, abs(x)), step)
                            x_new = x - step
                            if not _fin(x_new):
                                break
                            x = x_new
                        continue
            
            # Also try using the first deg+1 points in order
            if len(points) >= deg + 1:
                xs2 = [p[0] for p in points[:deg + 1]]
                fs2 = [p[1] for p in points[:deg + 1]]
                root_est2 = _poly_interp_root(xs2, fs2, degree=deg)
                if root_est2 is not None and _fin(root_est2) and root_est2 != root_est:
                    fx_root2 = _eval(root_est2)
                    if fx_root2 is not None:
                        if abs(fx_root2) <= 1e-8:
                            return root_est2
                        if abs(fx_root2) < 1e-4:
                            x = root_est2
                            for i in range(15):
                                if call_count >= _CALL_LIMIT or _left() < 1.0:
                                    break
                                fx = _eval(x)
                                if fx is None:
                                    break
                                if abs(fx) <= 1e-8:
                                    return x
                                h = 1e-6 * max(1.0, abs(x))
                                fxh = _eval(x + h)
                                if fxh is None:
                                    break
                                d = (fxh - fx) / h
                                if not _fin(d) or abs(d) < 1e-15:
                                    break
                                step = fx / d
                                if abs(step) > 5.0 * max(1.0, abs(x)):
                                    step = math.copysign(5.0 * max(1.0, abs(x)), step)
                                x_new = x - step
                                if not _fin(x_new):
                                    break
                                x = x_new
    
    # Strategy 2: Newton's method with adaptive step and damping
    if call_count < _CALL_LIMIT and _left() > 2.0 and abs(best_fx) > 1e-8:
        x = best_x
        for i in range(40):
            if call_count >= _CALL_LIMIT or _left() < 2.0:
                break
            fx = _eval(x)
            if fx is None:
                break
            if abs(fx) <= 1e-8:
                return x
            
            # Finite difference derivative with adaptive h
            h = 1e-5 * max(1.0, abs(x))
            fxh = _eval(x + h)
            if fxh is None:
                h = 1e-7 * max(1.0, abs(x))
                fxh = _eval(x + h)
                if fxh is None:
                    break
            d = (fxh - fx) / h
            if not _fin(d) or abs(d) < 1e-15:
                break
            
            # Newton step with damping if step is too large
            step = fx / d
            if abs(step) > 10.0 * max(1.0, abs(x)):
                step = math.copysign(10.0 * max(1.0, abs(x)), step)
            x_new = x - step
            if not _fin(x_new):
                break
            
            fxn = _eval(x_new)
            if fxn is None:
                # Damp the step further
                x_new = x - 0.5 * step
                if not _fin(x_new):
                    break
                fxn = _eval(x_new)
                if fxn is None:
                    break
            
            if abs(fxn) <= 1e-8:
                return x_new
            
            # If the step didn't help, damp it
            if abs(fxn) > abs(fx):
                x_new = x - 0.5 * step
                if not _fin(x_new):
                    break
                fxn = _eval(x_new)
                if fxn is None:
                    break
                if abs(fxn) <= 1e-8:
                    return x_new
                if abs(fxn) > abs(fx):
                    break
            
            x = x_new
    
    # Strategy 3: Secant method
    if call_count < _CALL_LIMIT and _left() > 2.0 and abs(best_fx) > 1e-8:
        x1 = x0
        x2 = x0 + 0.5 * max(1.0, abs(x0))
        f1 = _eval(x1)
        f2 = _eval(x2)
        
        if f1 is not None and abs(f1) <= 1e-8:
            return x1
        if f2 is not None and abs(f2) <= 1e-8:
            return x2
        
        if f1 is not None and f2 is not None and f2 != f1:
            for i in range(30):
                if call_count >= _CALL_LIMIT or _left() < 2.0:
                    break
                x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
                if not _fin(x_new):
                    break
                f_new = _eval(x_new)
                if f_new is None:
                    break
                if abs(f_new) <= 1e-8:
                    return x_new
                x1, f1 = x2, f2
                x2, f2 = x_new, f_new
    
    # Strategy 4: Bisection with bracket expansion
    if call_count < _CALL_LIMIT and _left() > 2.0 and abs(best_fx) > 1e-8:
        for scale in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
            if call_count >= _CALL_LIMIT or _left() < 2.0:
                break
            a = x0 - scale
            b = x0 + scale
            fa = _eval(a)
            fb = _eval(b)
            
            if fa is not None and abs(fa) <= 1e-8:
                return a
            if fb is not None and abs(fb) <= 1e-8:
                return b
            
            if fa is not None and fb is not None and fa * fb < 0:
                for i in range(60):
                    if call_count >= _CALL_LIMIT or _left() < 1.0:
                        break
                    mid = (a + b) / 2.0
                    fm = _eval(mid)
                    if fm is None:
                        break
                    if abs(fm) <= 1e-8:
                        return mid
                    if fa * fm < 0:
                        b, fb = mid, fm
                    else:
                        a, fa = mid, fm
                    if abs(b - a) < 1e-14:
                        break
                break
    
    # Strategy 5: Local minimization of |f(x)| using a simple gradient-free approach
    if call_count < _CALL_LIMIT and _left() > 2.0 and abs(best_fx) > 1e-8:
        x = best_x
        step = 0.1 * max(1.0, abs(x))
        for i in range(20):
            if call_count >= _CALL_LIMIT or _left() < 1.0:
                break
            best_local = (x, abs(best_fx))
            for dx in [-step, step, -0.5*step, 0.5*step, -2*step, 2*step]:
                if call_count >= _CALL_LIMIT:
                    break
                x_try = x + dx
                if not _fin(x_try):
                    continue
                fx_try = _eval(x_try)
                if fx_try is None:
                    continue
                if abs(fx_try) <= 1e-8:
                    return x_try
                if abs(fx_try) < best_local[1]:
                    best_local = (x_try, abs(fx_try))
            if best_local[0] == x:
                step *= 0.5
                if step < 1e-10:
                    break
            else:
                x = best_local[0]
    
    return best_x
# EVOLVE-BLOCK-END
