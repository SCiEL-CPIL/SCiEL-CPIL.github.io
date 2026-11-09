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
        return mid, fmid

    def _secant_step(x, fx, x_prev, fx_prev):
        """Secant method step."""
        if fx_prev == fx:
            return None
        x_new = x - fx * (x - x_prev) / (fx - fx_prev)
        if not math.isfinite(x_new):
            return None
        return x_new

    def _newton_deriv_forward(x, fx):
        """Compute derivative via central difference (2 extra evals) with forward fallback."""
        scale = max(1.0, abs(x))
        for h_factor in [1e-5, 1e-4]:
            h = h_factor * scale
            if h == 0:
                continue
            fp = _f(x + h)
            fm = _f(x - h)
            if math.isfinite(fp) and math.isfinite(fm):
                d = (fp - fm) / (2.0 * h)
                if math.isfinite(d) and abs(d) > 1e-15:
                    return d
            elif math.isfinite(fp):
                d = (fp - fx) / h
                if math.isfinite(d) and abs(d) > 1e-15:
                    return d
        return None

    def _newton_step(x, fx):
        d = _newton_deriv_forward(x, fx)
        if d is None:
            return None
        step = fx / d
        max_step = 5.0 * max(1.0, abs(x))
        if abs(step) > max_step:
            step = max_step if step > 0 else -max_step
        x_new = x - step
        if not math.isfinite(x_new):
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

    for i in range(120):
        if not _budget_ok():
            break
        if abs(fx) <= 1e-8:
            return x
        # Secant first (1 eval/step), Newton as fallback (2 evals/step)
        x_new = _secant_step(x, fx, x_prev, fx_prev)
        if x_new is None:
            x_new = _newton_step(x, fx)
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
        for mult in [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 2000.0]:
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

    # --- Phase 3: Cheap Newton polish from best_x (central-diff), target 1e-12 ---
    if _budget_ok() and best_abs > 1e-8:
        x = best_x
        fx = best_fx
        for _ in range(40):
            if not _budget_ok():
                break
            if abs(fx) <= 1e-12:
                return x
            x_new = _newton_step(x, fx)
            if x_new is None:
                break
            fx_new = _f(x_new)
            _update(x_new, fx_new)
            if not math.isfinite(fx_new):
                break
            if abs(fx_new) > abs(fx):
                break
            x, fx = x_new, fx_new
            if abs(x - x_new) < 1e-15 * max(1.0, abs(x)):
                break
        # If we got to 1e-12, return; otherwise keep going to Phase 4
        if math.isfinite(fx) and abs(fx) <= 1e-12:
            return x

    # --- Phase 4: Secant polish, target 1e-12 ---
    if _budget_ok() and best_abs > 1e-8:
        x = best_x
        fx = best_fx
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

        for _ in range(60):
            if not _budget_ok():
                break
            if abs(fx) <= 1e-12:
                return x
            if fx == fx_prev:
                break
            x_new = x - fx * (x - x_prev) / (fx - fx_prev)
            if not math.isfinite(x_new):
                break
            fx_new = _f(x_new)
            _update(x_new, fx_new)
            if not math.isfinite(fx_new):
                break
            if abs(fx_new) <= 1e-12:
                return x_new
            if abs(fx_new) < abs(fx):
                x_prev, fx_prev = x, fx
                x, fx = x_new, fx_new
            else:
                x_prev, fx_prev = x_new, fx_new
            if abs(x - x_prev) < 1e-15 * max(1.0, abs(x)):
                break

    return best_x
