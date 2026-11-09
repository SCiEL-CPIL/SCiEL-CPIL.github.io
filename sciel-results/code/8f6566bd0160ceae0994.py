"""Direct root-finder: secant-primary with bracket-bisect and cheap Newton polish."""
import time
import math

def find_root(f, x0):
    _t0 = time.time()
    _DEADLINE = 145.0  # safety margin before 150s
    _MAX_EVALS = 490
    _evals = 0

    def _f(x):
        nonlocal _evals
        _evals += 1
        try:
            val = f(x)
            return float(val)
        except Exception:
            return float('nan')

    def _time_ok():
        return (time.time() - _t0) < _DEADLINE

    def _budget_ok():
        return _evals < _MAX_EVALS and _time_ok()

    # Best candidate
    best_x = float(x0)
    best_fx = _f(best_x)
    best_abs = abs(best_fx) if math.isfinite(best_fx) else float('inf')

    def _update(x, fx):
        nonlocal best_x, best_fx, best_abs
        if not math.isfinite(fx):
            return
        af = abs(fx)
        if af < best_abs:
            best_abs = af
            best_x = float(x)
            best_fx = float(fx)

    _update(best_x, best_fx)
    if best_abs <= 1e-8:
        return best_x

    def _bisection(a, b, fa, fb):
        """Bisect on [a,b] where fa*fb <= 0. Returns (best_x, best_f) found."""
        nonlocal best_x, best_fx, best_abs
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        for _ in range(200):
            if not _budget_ok():
                break
            mid = (a + b) / 2.0
            if mid == a or mid == b:
                break
            fmid = _f(mid)
            _update(mid, fmid)
            if not math.isfinite(fmid):
                break
            if abs(fmid) <= 1e-8:
                return mid, fmid
            if (fmid <= 0 and fa <= 0) or (fmid >= 0 and fa >= 0):
                a, fa = mid, fmid
            else:
                b, fb = mid, fmid
            if abs(b - a) < 1e-15 * max(1.0, abs(a), abs(b)):
                break
        # Final secant polish if we have a bracket with distinct function values
        if math.isfinite(fa) and math.isfinite(fb) and fa != fb:
            x_new = a - fa * (a - b) / (fa - fb)
            if math.isfinite(x_new):
                fx_new = _f(x_new)
                _update(x_new, fx_new)
                if math.isfinite(fx_new) and abs(fx_new) <= 1e-8:
                    return x_new, fx_new
        return mid, fmid

    def _secant_step(x, fx, x_prev, fx_prev):
        """Secant method step."""
        if fx_prev == fx:
            return None
        x_new = x - fx * (x - x_prev) / (fx - fx_prev)
        if not math.isfinite(x_new):
            return None
        return x_new

    def _iqi_step(x1, f1, x2, f2, x3, f3):
        """Inverse quadratic interpolation step using last 3 points.
        Returns x_new or None if not applicable.
        """
        if not (math.isfinite(x1) and math.isfinite(x2) and math.isfinite(x3)):
            return None
        if not (math.isfinite(f1) and math.isfinite(f2) and math.isfinite(f3)):
            return None
        # Need distinct x values
        if x1 == x2 or x1 == x3 or x2 == x3:
            return None
        # f2 should be the smallest |f| for IQU to work well
        # But we can still try; use standard formula
        # Interpolate x as function of f (inverse)
        # x(f) = x1 * L1(f) + x2 * L2(f) + x3 * L3(f) where Li are Lagrange in f
        # We want f=0
        # x(0) = x1 * (f2*(f3)) / ((f1-f2)*(f1-f3)) + x2 * (f1*(f3)) / ((f2-f1)*(f2-f3)) + x3 * (f1*(f2)) / ((f3-f1)*(f3-f2))
        # Wait, let me use the standard IQI formula:
        # x_new = x1 + f1 * (f2-f1)/(f1-f2) ... no
        # Standard: fit quadratic x = A*f^2 + B*f + C through (f1,x1),(f2,x2),(f3,x3)
        # x(0) = C
        # Using Lagrange:
        # C = x1 * f2*f3/((f1-f2)*(f1-f3)) + x2 * f1*f3/((f2-f1)*(f2-f3)) + x3 * f1*f2/((f3-f1)*(f3-f2))
        d12 = f1 - f2
        d13 = f1 - f3
        d23 = f2 - f3
        if abs(d12) < 1e-30 or abs(d13) < 1e-30 or abs(d23) < 1e-30:
            return None
        x_new = (x1 * f2 * f3 / (d12 * d13) +
                 x2 * f1 * f3 / (-d12 * d23) +
                 x3 * f1 * f2 / (d13 * d23))
        if not math.isfinite(x_new):
            return None
        # Sanity: don't go too far
        max_dist = 10.0 * max(1.0, abs(x1), abs(x2), abs(x3))
        if abs(x_new - x1) > max_dist:
            return None
        return x_new

    # --- Phase 1: Secant-primary with bracket tracking ---
    x = float(x0)
    fx = _f(x)
    _update(x, fx)
    if math.isfinite(fx) and abs(fx) <= 1e-8:
        return x

    bracket_a = bracket_b = None
    bracket_fa = bracket_fb = None
    x_prev = x + 1e-6 * max(1.0, abs(x))
    fx_prev = _f(x_prev)
    _update(x_prev, fx_prev)

    # Maintain a history of last 3 points for IQI
    hist_x = [x, x_prev]
    hist_f = [fx, fx_prev]

    for i in range(120):
        if not _budget_ok():
            break
        if abs(fx) <= 1e-8:
            return x
        # Try IQI first if we have 3 points, else secant
        x_new = None
        if len(hist_x) >= 3:
            x_new = _iqi_step(hist_x[-3], hist_f[-3], hist_x[-2], hist_f[-2], hist_x[-1], hist_f[-1])
        if x_new is None:
            x_new = _secant_step(x, fx, x_prev, fx_prev)
        if x_new is None:
            break
        fx_new = _f(x_new)
        _update(x_new, fx_new)
        if not math.isfinite(fx_new):
            break
        if abs(fx_new) <= 1e-8:
            return x_new
        # Track bracket on sign change
        if (fx_new <= 0 and fx >= 0) or (fx_new >= 0 and fx <= 0):
            if abs(fx) <= abs(fx_new):
                bracket_a, bracket_b = x, x_new
                bracket_fa, bracket_fb = fx, fx_new
            else:
                bracket_a, bracket_b = x_new, x
                bracket_fb, bracket_fa = fx, fx_new
        # Update history (keep last 3)
        hist_x.append(x_new)
        hist_f.append(fx_new)
        if len(hist_x) > 3:
            hist_x.pop(0)
            hist_f.pop(0)
        x_prev, fx_prev = x, fx
        x, fx = x_new, fx_new
        if abs(x - x_prev) < 1e-14 * max(1.0, abs(x)):
            break

    if abs(fx) <= 1e-8 and math.isfinite(fx):
        return x
    _update(x, fx)

    # If we have a bracket, bisect
    if bracket_a is not None and bracket_b is not None:
        if (bracket_fa <= 0 and bracket_fb >= 0) or (bracket_fa >= 0 and bracket_fb <= 0):
            mid, fmid = _bisection(bracket_a, bracket_b, bracket_fa, bracket_fb)
            _update(mid, fmid)
            if abs(fmid) <= 1e-8:
                return mid

    # --- Phase 2: Scan for a bracket around x0 ---
    if _budget_ok() and best_abs > 1e-8:
        base = float(x0)
        scan_scale = max(1.0, abs(base))
        for mult in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0]:
            if not _budget_ok():
                break
            for sign in [-1, 1]:
                if not _budget_ok():
                    break
                a = base - sign * mult * scan_scale
                b = base + sign * mult * scan_scale
                if a == b:
                    continue
                fa = _f(a)
                fb = _f(b)
                _update(a, fa)
                _update(b, fb)
                if not (math.isfinite(fa) and math.isfinite(fb)):
                    continue
                if abs(fa) <= 1e-8:
                    return a
                if abs(fb) <= 1e-8:
                    return b
                if (fa <= 0 and fb >= 0) or (fa >= 0 and fb <= 0):
                    mid, fmid = _bisection(a, b, fa, fb)
                    _update(mid, fmid)
                    if abs(fmid) <= 1e-8:
                        return mid
                    break
            else:
                continue
            break

    # --- Phase 3: Secant polish from best_x (no derivative needed) ---
    if _budget_ok() and best_abs > 1e-8:
        x = best_x
        fx = _f(x)
        _update(x, fx)
        if not math.isfinite(fx):
            return best_x
        h = 1e-6 * max(1.0, abs(x))
        x_prev = x + h
        fx_prev = _f(x_prev)
        _update(x_prev, fx_prev)
        if not math.isfinite(fx_prev) or fx_prev == fx:
            x_prev = x - h
            fx_prev = _f(x_prev)
            _update(x_prev, fx_prev)
            if not math.isfinite(fx_prev) or fx_prev == fx:
                return best_x

        # IQI history for polish
        hist_x = [x, x_prev]
        hist_f = [fx, fx_prev]

        for _ in range(40):
            if not _budget_ok():
                break
            if abs(fx) <= 1e-8:
                return x
            x_new = None
            if len(hist_x) >= 3:
                x_new = _iqi_step(hist_x[-3], hist_f[-3], hist_x[-2], hist_f[-2], hist_x[-1], hist_f[-1])
            if x_new is None:
                x_new = _secant_step(x, fx, x_prev, fx_prev)
            if x_new is None:
                break
            fx_new = _f(x_new)
            _update(x_new, fx_new)
            if not math.isfinite(fx_new):
                break
            if abs(fx_new) > abs(fx):
                # Accept anyway for history but stop if no progress
                break
            x_prev, fx_prev = x, fx
            x, fx = x_new, fx_new
            hist_x.append(x_new)
            hist_f.append(fx_new)
            if len(hist_x) > 3:
                hist_x.pop(0)
                hist_f.pop(0)
            if abs(x - x_new) < 1e-15 * max(1.0, abs(x)):
                break

    # Phase 3 already does secant/IQI polish; Phase 4 removed to save evals
    return best_x
