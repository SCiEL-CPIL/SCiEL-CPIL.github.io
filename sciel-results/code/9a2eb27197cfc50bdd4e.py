"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

_DEADLINE_START = time.time()
_WALL_LIMIT = 115.0  # safety margin before 150s kill

def _time_left():
    return _WALL_LIMIT - (time.time() - _DEADLINE_START)

def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)

def find_root(f, x0):
    x = float(x0)
    best_x = x
    best_fx = None
    try:
        best_fx = f(x)
        if _is_finite(best_fx):
            best_x = x
    except Exception:
        pass
    
    if best_fx is not None and _is_finite(best_fx) and abs(best_fx) <= 1e-8:
        return best_x
    
    # Try Newton's method with finite-difference derivative
    x_newton = x
    for i in range(50):
        if _time_left() < 1.0:
            break
        try:
            fx = f(x_newton)
        except Exception:
            break
        if not _is_finite(fx):
            break
        if _is_finite(best_fx) is False or abs(fx) < abs(best_fx):
            best_x, best_fx = x_newton, fx
        if abs(fx) <= 1e-8:
            return x_newton
        h = 1e-6 * max(1.0, abs(x_newton))
        try:
            fx_ph = f(x_newton + h)
        except Exception:
            break
        if not _is_finite(fx_ph):
            break
        d = (fx_ph - fx) / h
        if not _is_finite(d) or d == 0:
            break
        x_newton = x_newton - fx / d
        if not _is_finite(x_newton):
            break
    
    # If Newton didn't converge well, try secant method from two nearby points
    if _time_left() > 1.0 and (best_fx is None or abs(best_fx) > 1e-8):
        x1 = x
        x2 = x + 0.1 * max(1.0, abs(x))
        try:
            f1 = f(x1)
            f2 = f(x2)
        except Exception:
            f1 = f2 = None
        if f1 is not None and f2 is not None and _is_finite(f1) and _is_finite(f2):
            if abs(f1) < abs(best_fx if _is_finite(best_fx) else float('inf')):
                best_x, best_fx = x1, f1
            if abs(f2) < abs(best_fx if _is_finite(best_fx) else float('inf')):
                best_x, best_fx = x2, f2
            if abs(f1) <= 1e-8:
                return x1
            if abs(f2) <= 1e-8:
                return x2
            
            for i in range(50):
                if _time_left() < 1.0:
                    break
                if f2 - f1 == 0:
                    break
                x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
                if not _is_finite(x_new):
                    break
                try:
                    f_new = f(x_new)
                except Exception:
                    break
                if not _is_finite(f_new):
                    break
                if abs(f_new) < abs(best_fx if _is_finite(best_fx) else float('inf')):
                    best_x, best_fx = x_new, f_new
                if abs(f_new) <= 1e-8:
                    return x_new
                x1, f1 = x2, f2
                x2, f2 = x_new, f_new
    
    # If still not good enough, try bisection around x0
    if _time_left() > 1.0 and (best_fx is None or abs(best_fx) > 1e-8):
        # Try to find a bracket
        a, b = x - 1.0, x + 1.0
        fa, fb = None, None
        try:
            fa = f(a)
        except Exception:
            pass
        try:
            fb = f(b)
        except Exception:
            pass
        if fa is not None and fb is not None and _is_finite(fa) and _is_finite(fb):
            if abs(fa) < abs(best_fx if _is_finite(best_fx) else float('inf')):
                best_x, best_fx = a, fa
            if abs(fb) < abs(best_fx if _is_finite(best_fx) else float('inf')):
                best_x, best_fx = b, fb
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
            # Expand bracket if no sign change
            if fa * fb > 0:
                for scale in [2, 4, 8, 16, 32, 64, 128, 256]:
                    if _time_left() < 1.0:
                        break
                    a_new = x - scale
                    b_new = x + scale
                    try:
                        fa_new = f(a_new)
                        fb_new = f(b_new)
                    except Exception:
                        continue
                    if not _is_finite(fa_new) or not _is_finite(fb_new):
                        continue
                    if abs(fa_new) < abs(best_fx if _is_finite(best_fx) else float('inf')):
                        best_x, best_fx = a_new, fa_new
                    if abs(fb_new) < abs(best_fx if _is_finite(best_fx) else float('inf')):
                        best_x, best_fx = b_new, fb_new
                    if abs(fa_new) <= 1e-8:
                        return a_new
                    if abs(fb_new) <= 1e-8:
                        return b_new
                    if fa_new * fb_new < 0:
                        a, b = a_new, b_new
                        fa, fb = fa_new, fb_new
                        break
            # Bisection if bracket found
            if fa * fb < 0:
                for i in range(60):
                    if _time_left() < 1.0:
                        break
                    mid = (a + b) / 2.0
                    try:
                        fm = f(mid)
                    except Exception:
                        break
                    if not _is_finite(fm):
                        break
                    if abs(fm) < abs(best_fx if _is_finite(best_fx) else float('inf')):
                        best_x, best_fx = mid, fm
                    if abs(fm) <= 1e-8:
                        return mid
                    if fa * fm < 0:
                        b, fb = mid, fm
                    else:
                        a, fa = mid, fm
                    if abs(b - a) < 1e-12:
                        break
    
    # Final high-precision polish pass using Newton with tighter tolerance
    if best_fx is not None and _is_finite(best_fx) and abs(best_fx) > 1e-8:
        x_polish = best_x
        for i in range(30):
            if _time_left() < 0.5:
                break
            try:
                fx = f(x_polish)
            except Exception:
                break
            if not _is_finite(fx):
                break
            if abs(fx) < abs(best_fx):
                best_x, best_fx = x_polish, fx
            if abs(fx) <= 1e-8:
                return x_polish
            h = 1e-8 * max(1.0, abs(x_polish))
            try:
                fx_ph = f(x_polish + h)
            except Exception:
                break
            if not _is_finite(fx_ph):
                break
            d = (fx_ph - fx) / h
            if not _is_finite(d) or d == 0:
                break
            x_polish = x_polish - fx / d
            if not _is_finite(x_polish):
                break
    
    return best_x
# EVOLVE-BLOCK-END
