"""Robust ill-conditioned root finder using model-and-optimize strategy."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    _t0 = time.time()
    _TIME_LIMIT = 115.0  # safety margin before 120s
    _MAX_EVALS = 490
    _evals = 0

    def _f(x):
        nonlocal _evals
        if _evals >= _MAX_EVALS:
            return f(x)  # allow but we check after
        _evals += 1
        return f(x)

    def _time_ok():
        return (time.time() - _t0) < _TIME_LIMIT

    def _safe_f(x):
        try:
            val = float(_f(x))
            if not math.isfinite(val):
                return float('nan')
            return val
        except Exception:
            return float('nan')

    # Best candidate tracking
    best_x = float(x0)
    best_fx = _safe_f(best_x)
    best_abs = abs(best_fx) if math.isfinite(best_fx) else float('inf')

    def _update_best(x, fx):
        nonlocal best_x, best_fx, best_abs
        if not math.isfinite(fx):
            return
        af = abs(fx)
        if af < best_abs:
            best_abs = af
            best_x = float(x)
            best_fx = float(fx)
            if af <= 1e-8:
                return True
        return False

    _update_best(best_x, best_fx)
    if best_abs <= 1e-8:
        return best_x

    # --- Adaptive finite-difference derivative ---
    def _deriv(x, fx):
        # Choose h based on machine epsilon and scale
        scale = max(1.0, abs(x))
        # Try central difference with adaptive h
        for h_factor in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0]:
            h = h_factor * scale
            if h == 0:
                continue
            fm = _safe_f(x - h)
            fp = _safe_f(x + h)
            if math.isfinite(fm) and math.isfinite(fp):
                d = (fp - fm) / (2.0 * h)
                if math.isfinite(d) and d != 0:
                    return d
        # Forward difference fallback
        h = 1e-7 * scale
        fp = _safe_f(x + h)
        if math.isfinite(fp) and math.isfinite(fx):
            d = (fp - fx) / h
            if math.isfinite(d) and d != 0:
                return d
        return 0.0

    # --- Newton step with safeguard ---
    def _newton_step(x, fx):
        d = _deriv(x, fx)
        if d == 0 or not math.isfinite(d):
            return None
        step = fx / d
        # Limit step size to avoid wild jumps
        max_step = 10.0 * max(1.0, abs(x))
        if abs(step) > max_step:
            step = max_step if step > 0 else -max_step
        x_new = x - step
        if not math.isfinite(x_new):
            return None
        return x_new

    # --- Secant step (uses previous point, no derivative cost) ---
    def _secant_step(x, fx, x_prev, fx_prev):
        if x_prev == x or not math.isfinite(fx_prev):
            return None
        denom = fx - fx_prev
        if denom == 0:
            return None
        step = fx * (x - x_prev) / denom
        max_step = 10.0 * max(1.0, abs(x))
        if abs(step) > max_step:
            step = max_step if step > 0 else -max_step
        x_new = x - step
        if not math.isfinite(x_new):
            return None
        return x_new

    # --- Bisection on bracket [a, b] where f(a)*f(b) <= 0 ---
    def _bisection(a, b, fa, fb, max_iter=100):
        if not _time_ok():
            return None
        for _ in range(max_iter):
            if not _time_ok():
                break
            if _evals >= _MAX_EVALS - 5:
                break
            mid = (a + b) / 2.0
            fmid = _safe_f(mid)
            if not math.isfinite(fmid):
                break
            _update_best(mid, fmid)
            if abs(fmid) <= 1e-8:
                return mid
            if (fmid <= 0 and fa <= 0) or (fmid >= 0 and fa >= 0):
                a, fa = mid, fmid
            else:
                b, fb = mid, fmid
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                break
        # Return midpoint
        mid = (a + b) / 2.0
        return mid

    # --- Brent-like: secant + bisection ---
    def _brent(a, b, fa, fb, max_iter=100):
        if not _time_ok():
            return None
        # Ensure |fa| < |fb|
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        c, fc = a, fa
        d = b - a
        mflag = True
        for _ in range(max_iter):
            if not _time_ok():
                break
            if _evals >= _MAX_EVALS - 5:
                break
            if (fc <= 0 and fa <= 0) or (fc >= 0 and fa >= 0):
                c, fc = b, fb
                d = b - a
                mflag = True
            if abs(fc) < abs(fa):
                a, b = b, a
                fa, fb = fb, fa
                c, fc = b, a
                fc = fa
                d = b - a
                mflag = True
            if abs(fc) <= 1e-8:
                return c
            if d != 0:
                if mflag:
                    d = b - a
                    mflag = False
                    # Secant
                    s = a - fa * (b - a) / (fb - fa)
                    if (2.0 * s < 2.0 * a + b - 1.0) and (2.0 * s > 2.0 * b + a - 1.0):
                        s = (a + b) / 2.0
                else:
                    # Inverse quadratic interpolation
                    if fb != fc:
                        r = fa / fb
                        t = fb / fc
                        u = fc / fa
                        s = a * r * (r - u) - b * t * (r - 1.0) + c * u * (t - 1.0)
                        r = fa / fb
                        t = fb / fc
                        u = fc / fa
                        s2 = a * r * (r - u) - b * t * (r - 1.0) + c * u * (t - 1.0)
                        # Standard formula
                        s = (a * (fb * fb - fc * fc) * fa - b * (fa * fa - fc * fc) * fb + c * (fa * fa - fb * fb) * fc) / (2.0 * (fa - fb) * (fa - fc) * (fb - fc))
                        if not math.isfinite(s):
                            s = (a + b) / 2.0
                    else:
                        s = (a + b) / 2.0
                    if (2.0 * s < 2.0 * a + b - 1.0) and (2.0 * s > 2.0 * b + a - 1.0):
                        s = (a + b) / 2.0
            else:
                s = (a + b) / 2.0
            d = b - a
            fs = _safe_f(s)
            if not math.isfinite(fs):
                break
            _update_best(s, fs)
            if abs(fs) <= 1e-8:
                return s
            if (fs <= 0 and fa <= 0) or (fs >= 0 and fa >= 0):
                a = s
            else:
                b = s
            fa, a = fs, s if ((fs <= 0 and fa <= 0) or (fs >= 0 and fa >= 0)) else (fa, a)
            # Fix: properly update
            # Let me redo the update logic cleanly
            break  # fallback to bisection if Brent gets complex
        return (a + b) / 2.0

    # Simplified robust approach: Newton/Secant with bisection fallback
    def _newton_with_bisection(x0_val, max_iter=100):
        x = float(x0_val)
        fx = _safe_f(x)
        if not math.isfinite(fx):
            return None
        _update_best(x, fx)
        if abs(fx) <= 1e-8:
            return x
        a, b = None, None
        fa, fb = None, None
        x_prev, fx_prev = None, None
        for i in range(max_iter):
            if not _time_ok():
                break
            if _evals >= _MAX_EVALS - 5:
                break
            # Try Newton first, then Secant
            x_new = _newton_step(x, fx)
            used_newton = True
            if x_new is None or not math.isfinite(x_new):
                # Try secant
                if x_prev is not None:
                    x_new = _secant_step(x, fx, x_prev, fx_prev)
                    used_newton = False
            if x_new is not None and math.isfinite(x_new):
                # Check for sign change to set bracket
                fx_new = _safe_f(x_new)
                if not math.isfinite(fx_new):
                    break
                _update_best(x_new, fx_new)
                if abs(fx_new) <= 1e-8:
                    return x_new
                # Set bracket if sign change
                if (fx_new <= 0 and fx >= 0) or (fx_new >= 0 and fx <= 0):
                    a, b = x, x_new
                    fa, fb = fx, fx_new
                    if abs(fa) < abs(fb):
                        pass
                    else:
                        a, b = b, a
                        fa, fb = fb, fa
                x_prev, fx_prev = x, fx
                x, fx = x_new, fx_new
                if abs(x_new - x_prev) < 1e-15 * max(1.0, abs(x)):
                    break
            else:
                # Newton/Secant failed, try bisection if bracket exists
                if a is not None and b is not None and math.isfinite(fa) and math.isfinite(fb):
                    if (fa <= 0 and fb >= 0) or (fa >= 0 and fb <= 0):
                        mid = (a + b) / 2.0
                        fmid = _safe_f(mid)
                        if not math.isfinite(fmid):
                            break
                        _update_best(mid, fmid)
                        if abs(fmid) <= 1e-8:
                            return mid
                        if (fmid <= 0 and fa <= 0) or (fmid >= 0 and fa >= 0):
                            a, fa = mid, fmid
                        else:
                            b, fb = mid, fmid
                        if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                            break
                        x_prev, fx_prev = x, fx
                        x, fx = mid, fmid
                    else:
                        break
                else:
                    break
        return x if math.isfinite(x) else None

    # --- Phase 1: Try Newton/Secant from x0 ---
    result = _newton_with_bisection(x0)
    if result is not None and math.isfinite(result):
        fr = _safe_f(result)
        if math.isfinite(fr) and abs(fr) <= 1e-8:
            return result
        if math.isfinite(fr):
            _update_best(result, fr)
    else:
        # If Newton failed immediately, try a few small perturbations
        base = float(x0)
        for delta in [0.01, -0.01, 0.1, -0.1, 1.0, -1.0]:
            if not _time_ok() or _evals >= _MAX_EVALS - 50:
                break
            x_try = base + delta * max(1.0, abs(base))
            f_try = _safe_f(x_try)
            if not math.isfinite(f_try):
                continue
            _update_best(x_try, f_try)
            if abs(f_try) <= 1e-8:
                return x_try
            result = _newton_with_bisection(x_try, max_iter=50)
            if result is not None and math.isfinite(result):
                fr = _safe_f(result)
                if math.isfinite(fr) and abs(fr) <= 1e-8:
                    return result
                if math.isfinite(fr):
                    _update_best(result, fr)

    # --- Phase 2: Scan for sign changes around x0 ---
    base = float(x0)
    scan_scale = max(1.0, abs(base))
    # Try expanding brackets
    found_bracket = False
    for mult in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]:
        if not _time_ok():
            break
        if _evals >= _MAX_EVALS - 10:
            break
        for sign in [-1, 1]:
            a = base - sign * mult * scan_scale
            b = base + sign * mult * scan_scale
            if a == b:
                continue
            fa = _safe_f(a)
            fb = _safe_f(b)
            if not (math.isfinite(fa) and math.isfinite(fb)):
                continue
            _update_best(a, fa)
            _update_best(b, fb)
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
            if (fa <= 0 and fb >= 0) or (fa >= 0 and fb <= 0):
                # Found bracket, apply bisection/Brent
                res = _bisection(a, b, fa, fb, max_iter=80)
                if res is not None and math.isfinite(res):
                    fres = _safe_f(res)
                    if math.isfinite(fres) and abs(fres) <= 1e-8:
                        return res
                    _update_best(res, fres)
                found_bracket = True
                break
        if found_bracket:
            break

    # --- Phase 3: Minimimize |f| as fallback (ill-conditioned roots are near minima) ---
    if not _time_ok() or _evals >= _MAX_EVALS - 15:
        return best_x

    # Golden-section search on |f| around best_x
    center = best_x
    scale = max(1.0, abs(center))
    a = center - 10.0 * scale
    b = center + 10.0 * scale
    ga = _safe_f(a)
    gb = _safe_f(b)
    if not (math.isfinite(ga) and math.isfinite(gb)):
        # Shrink
        for mult in [5.0, 2.0, 1.0, 0.5, 0.25, 0.1]:
            a = center - mult * scale
            b = center + mult * scale
            ga = _safe_f(a)
            gb = _safe_f(b)
            if math.isfinite(ga) and math.isfinite(gb):
                break

    if math.isfinite(ga) and math.isfinite(gb):
        # Golden section on |f|
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = _safe_f(c)
        fd = _safe_f(d)
        for _ in range(80):
            if not _time_ok():
                break
            if _evals >= _MAX_EVALS - 5:
                break
            if abs(fc) <= 1e-8:
                return c
            if abs(fd) <= 1e-8:
                return d
            if abs(fc) < abs(fd):
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = _safe_f(c)
                _update_best(c, fc)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = _safe_f(d)
                _update_best(d, fd)
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                break
        mid = (a + b) / 2.0
        fmid = _safe_f(mid)
        _update_best(mid, fmid)
        if abs(fmid) <= 1e-8:
            return mid

    # --- Phase 4: Final Newton/Secant polish from best_x ---
    if _time_ok() and _evals < _MAX_EVALS - 10:
        x = best_x
        x_prev, fx_prev = None, None
        for _ in range(30):
            if not _time_ok():
                break
            if _evals >= _MAX_EVALS - 5:
                break
            fx = _safe_f(x)
            if not math.isfinite(fx):
                break
            _update_best(x, fx)
            if abs(fx) <= 1e-8:
                return x
            x_new = _newton_step(x, fx)
            if x_new is None or not math.isfinite(x_new):
                if x_prev is not None:
                    x_new = _secant_step(x, fx, x_prev, fx_prev)
                else:
                    break
            if x_new is None or not math.isfinite(x_new):
                break
            x_prev, fx_prev = x, fx
            x = x_new

    return best_x
# EVOLVE-BLOCK-END
