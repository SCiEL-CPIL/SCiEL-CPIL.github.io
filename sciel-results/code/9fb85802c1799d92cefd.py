"""Model-and-optimize root finder for ill-conditioned equations."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    _t0 = time.time()
    _TIME_LIMIT = 115.0  # safety margin before 120s
    _MAX_EVALS = 450
    _evals = 0

    def _f(x):
        nonlocal _evals
        _evals += 1
        try:
            return float(f(x))
        except Exception:
            return float('nan')

    def _time_ok():
        return (time.time() - _t0) < _TIME_LIMIT

    def _budget_ok():
        return _evals < _MAX_EVALS

    def _safe(x):
        return math.isfinite(x)

    # Best candidate tracking
    best_x = float(x0)
    best_fx = _f(best_x)
    best_abs = abs(best_fx) if _safe(best_fx) else float('inf')

    def _update(x, fx):
        nonlocal best_x, best_fx, best_abs
        if not _safe(fx):
            return False
        af = abs(fx)
        if af < best_abs:
            best_abs = af
            best_x = float(x)
            best_fx = float(fx)
            return af <= 1e-8
        return False

    _update(best_x, best_fx)
    if best_abs <= 1e-8:
        return best_x

    # Inverse quadratic interpolation (IQI)
    def _iqi(x1, f1, x2, f2, x3, f3):
        if not all(_safe(v) for v in [x1, f1, x2, f2, x3, f3]):
            return None
        denom = (f1 - f2) * (f1 - f3) * (f2 - f3)
        if abs(denom) < 1e-30:
            return None
        x0_iqi = (f2 * f3 * x1 / ((f1 - f2) * (f1 - f3)) +
                  f1 * f3 * x2 / ((f2 - f1) * (f2 - f3)) +
                  f1 * f2 * x3 / ((f3 - f1) * (f3 - f2)))
        if not _safe(x0_iqi):
            return None
        return x0_iqi

    # Secant method step
    def _secant(x1, f1, x2, f2):
        if not all(_safe(v) for v in [x1, f1, x2, f2]):
            return None
        denom = f2 - f1
        if abs(denom) < 1e-30:
            return None
        x_new = x1 - f1 * (x2 - x1) / denom
        if not _safe(x_new):
            return None
        return x_new

    # Bisection
    def _bisect(a, b, fa, fb, max_iter=60):
        for _ in range(max_iter):
            if not _time_ok() or not _budget_ok():
                break
            mid = (a + b) / 2.0
            fmid = _f(mid)
            if not _safe(fmid):
                break
            _update(mid, fmid)
            if abs(fmid) <= 1e-8:
                return mid
            if (fmid <= 0 and fa <= 0) or (fmid >= 0 and fa >= 0):
                a, fa = mid, fmid
            else:
                b, fb = mid, fmid
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                break
        return (a + b) / 2.0

    # --- Phase 1: IQI/Secant with bracket tracking ---
    x1 = float(x0)
    f1 = _f(x1)
    if not _safe(f1):
        x1 = best_x
        f1 = best_fx
    _update(x1, f1)

    if abs(f1) <= 1e-8:
        return x1

    # Try several offsets to find a second point (ideally with opposite sign)
    scale = max(1.0, abs(x1))
    offsets = [0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.01, -0.01]
    x2 = None
    f2 = None
    for off in offsets:
        if not _budget_ok():
            break
        cand = x1 + off * scale
        fc = _f(cand)
        if not _safe(fc):
            continue
        _update(cand, fc)
        if abs(fc) <= 1e-8:
            return cand
        # Prefer a point with opposite sign to f1
        if (f1 <= 0 and fc >= 0) or (f1 >= 0 and fc <= 0):
            x2, f2 = cand, fc
            break
        if x2 is None:
            x2, f2 = cand, fc
        elif abs(fc) < abs(f2):
            x2, f2 = cand, fc

    if x2 is None:
        x2 = x1 + 0.1 * scale
        f2 = _f(x2)
        _update(x2, f2)
        if not _safe(f2):
            x2 = x1 - 0.1 * scale
            f2 = _f(x2)
            _update(x2, f2)

    if abs(f2) <= 1e-8:
        return x2

    points = [(x1, f1), (x2, f2)]

    bracket_a = bracket_b = None
    bracket_fa = bracket_fb = None

    for i in range(80):
        if not _time_ok() or not _budget_ok():
            break

        x_prev = points[-1][0]
        f_prev = points[-1][1]

        if abs(f_prev) <= 1e-8:
            return x_prev

        x_new = None

        if len(points) >= 3:
            x3, f3 = points[-3]
            x2_, f2_ = points[-2]
            x1_, f1_ = points[-1]
            x_new = _iqi(x1_, f1_, x2_, f2_, x3, f3)

        if x_new is None and len(points) >= 2:
            x_a, f_a = points[-2]
            x_b, f_b = points[-1]
            x_new = _secant(x_a, f_a, x_b, f_b)

        if x_new is None:
            x_new = x_prev + 0.1 * max(1.0, abs(x_prev))

        max_step = 5.0 * max(1.0, abs(x_prev))
        if abs(x_new - x_prev) > max_step:
            x_new = x_prev + max_step * (1 if x_new > x_prev else -1)

        if not _safe(x_new):
            x_new = x_prev + 0.01 * max(1.0, abs(x_prev))
            if not _safe(x_new):
                break

        if any(abs(x_new - px) < 1e-15 * max(1.0, abs(px)) for px, _ in points):
            x_new = x_new + 1e-10 * max(1.0, abs(x_new))

        f_new = _f(x_new)
        if not _safe(f_new):
            x_new = x_prev - 0.5 * (x_new - x_prev)
            f_new = _f(x_new)
            if not _safe(f_new):
                break

        _update(x_new, f_new)
        if abs(f_new) <= 1e-8:
            return x_new

        if (f_new <= 0 and f_prev >= 0) or (f_new >= 0 and f_prev <= 0):
            bracket_a, bracket_b = x_prev, x_new
            bracket_fa, bracket_fb = f_prev, f_new
            if abs(bracket_fa) > abs(bracket_fb):
                bracket_a, bracket_b = bracket_b, bracket_a
                bracket_fa, bracket_fb = bracket_fb, bracket_fa

        points.append((x_new, f_new))
        if len(points) > 3:
            points.pop(0)

        if abs(x_new - x_prev) < 1e-15 * max(1.0, abs(x_new)):
            break

    # --- Phase 2: Bracket refinement with IQI acceleration ---
    if bracket_a is not None and bracket_b is not None:
        if (bracket_fa <= 0 and bracket_fb >= 0) or (bracket_fa >= 0 and bracket_fb <= 0):
            a, b = bracket_a, bracket_b
            fa, fb = bracket_fa, bracket_fb
            hist = [(a, fa), (b, fb)]

            for i in range(60):
                if not _time_ok() or not _budget_ok():
                    break

                x_trial = None
                if len(hist) >= 3:
                    x_trial = _iqi(hist[0][0], hist[0][1],
                                   hist[1][0], hist[1][1],
                                   hist[2][0], hist[2][1])

                if x_trial is None or not _safe(x_trial) or not (a < x_trial < b):
                    # Try secant on last two points
                    if len(hist) >= 2:
                        x_trial = _secant(hist[-2][0], hist[-2][1], hist[-1][0], hist[-1][1])
                        if x_trial is not None and _safe(x_trial) and (a < x_trial < b):
                            pass
                        else:
                            x_trial = (a + b) / 2.0
                    else:
                        x_trial = (a + b) / 2.0

                    if not _safe(x_trial) or not (a < x_trial < b):
                        x_trial = (a + b) / 2.0
                        if not _safe(x_trial):
                            break

                f_trial = _f(x_trial)
                if not _safe(f_trial):
                    break
                _update(x_trial, f_trial)
                if abs(f_trial) <= 1e-8:
                    return x_trial

                if (f_trial <= 0 and fa <= 0) or (f_trial >= 0 and fa >= 0):
                    a, fa = x_trial, f_trial
                else:
                    b, fb = x_trial, f_trial

                hist.append((x_trial, f_trial))
                if len(hist) > 3:
                    hist.pop(0)

                if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                    break

            res = (a + b) / 2.0
            fres = _f(res)
            _update(res, fres)
            if abs(fres) <= 1e-8:
                return res

    # --- Phase 3: Scan for brackets ---
    if not _time_ok() or not _budget_ok():
        return best_x

    base = best_x
    scan_scale = max(1.0, abs(base))
    found = False
    for mult in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]:
        if not _time_ok() or not _budget_ok():
            break
        for sign in [-1, 1]:
            a = base - sign * mult * scan_scale
            b = base + sign * mult * scan_scale
            if a == b:
                continue
            fa = _f(a)
            fb = _f(b)
            if not (_safe(fa) and _safe(fb)):
                continue
            _update(a, fa)
            _update(b, fb)
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
            if (fa <= 0 and fb >= 0) or (fa >= 0 and fb <= 0):
                res = _bisect(a, b, fa, fb, max_iter=50)
                fres = _f(res)
                _update(res, fres)
                if abs(fres) <= 1e-8:
                    return res
                found = True
                break
        if found:
            break

    # --- Phase 4: Minimize |f| via golden section ---
    if not _time_ok() or not _budget_ok():
        return best_x

    center = best_x
    scale = max(1.0, abs(center))
    a = center - 20.0 * scale
    b = center + 20.0 * scale

    fa = _f(a)
    fb = _f(b)
    if not (_safe(fa) and _safe(fb)):
        for mult in [10.0, 5.0, 2.0, 1.0, 0.5, 0.25, 0.1]:
            a = center - mult * scale
            b = center + mult * scale
            fa = _f(a)
            fb = _f(b)
            if _safe(fa) and _safe(fb):
                break
            a, b = center, center
            break

    if _safe(fa) and _safe(fb) and a != b:
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = _f(c)
        fd = _f(d)
        _update(c, fc)
        _update(d, fd)

        for _ in range(50):
            if not _time_ok() or not _budget_ok():
                break
            if abs(fc) <= 1e-8:
                return c
            if abs(fd) <= 1e-8:
                return d
            if abs(fc) < abs(fd):
                b, d, fd = d, c, fc
                c = b - gr * (b - a)
                fc = _f(c)
                _update(c, fc)
            else:
                a, c, fc = c, d, fd
                d = a + gr * (b - a)
                fd = _f(d)
                _update(d, fd)
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                break
        mid = (a + b) / 2.0
        fmid = _f(mid)
        _update(mid, fmid)
        if abs(fmid) <= 1e-8:
            return mid

    # --- Phase 5: Final secant polish with tighter tolerance ---
    if _time_ok() and _budget_ok():
        x_a = best_x
        f_a = _f(x_a)
        if not _safe(f_a):
            return best_x

        scale = max(1.0, abs(x_a))
        x_b = x_a + 0.01 * scale
        f_b = _f(x_b)
        if not _safe(f_b):
            x_b = x_a - 0.01 * scale
            f_b = _f(x_b)
            if not _safe(f_b):
                return best_x

        _update(x_a, f_a)
        _update(x_b, f_b)

        if abs(f_a) <= 1e-8:
            return x_a
        if abs(f_b) <= 1e-8:
            return x_b

        for _ in range(30):
            if not _time_ok() or not _budget_ok():
                break
            if abs(f_a) <= 1e-8:
                return x_a
            if abs(f_b) <= 1e-8:
                return x_b

            x_new = _secant(x_a, f_a, x_b, f_b)
            if x_new is None:
                # Try Newton-like step using derivative estimate
                if abs(x_b - x_a) > 1e-30:
                    deriv = (f_b - f_a) / (x_b - x_a)
                    if abs(deriv) > 1e-30:
                        x_new = x_a - f_a / deriv
                    else:
                        break
                else:
                    break

            max_step = 5.0 * max(1.0, abs(x_a))
            if abs(x_new - x_a) > max_step:
                x_new = x_a + max_step * (1 if x_new > x_a else -1)

            if not _safe(x_new):
                break

            f_new = _f(x_new)
            if not _safe(f_new):
                break

            _update(x_new, f_new)
            if abs(f_new) <= 1e-8:
                return x_new

            if abs(f_new) < abs(f_a):
                x_a, f_a = x_b, f_b
                x_b, f_b = x_new, f_new
            else:
                x_a, f_a = x_b, f_b
                x_b, f_b = x_new, f_new

    return best_x
# EVOLVE-BLOCK-END
