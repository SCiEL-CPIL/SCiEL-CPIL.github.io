"""Root finder for oscillatory equations: IQI primary with local secant/Newton polish and bracketing fallback."""
import math
import time

# Evaluator kills at 150s wall-clock; use 140s deadline for safety.
_DEADLINE = time.time() + 140.0
_CALL_BUDGET = 490


def _time_left():
    return _DEADLINE - time.time()


def find_root(f, x0):
    """Find x such that |f(x)| <= 1e-8 for an oscillatory function f."""
    counter = [0]
    max_calls = _CALL_BUDGET
    cache = {}

    def wf(x):
        """Counted evaluation with caching."""
        if x in cache:
            return cache[x][0]
        counter[0] += 1
        if counter[0] > max_calls:
            raise RuntimeError("call budget exceeded")
        try:
            v = f(x)
        except Exception:
            cache[x] = (float('nan'), False)
            return float('nan')
        ok = math.isfinite(v)
        cache[x] = (v, ok)
        return v

    best_x = float(x0)
    best_abs = float('inf')

    def try_update(x, val):
        nonlocal best_x, best_abs
        if math.isfinite(val):
            a = abs(val)
            if a < best_abs:
                best_x = x
                best_abs = a
                if best_abs <= 1e-8:
                    return True
        return False

    def safe_eval(x):
        """Evaluate wf(x), return (value, ok). ok=False if exception/non-finite."""
        v = wf(x)
        if not math.isfinite(v):
            return v, False
        return v, True

    x0f = float(x0)
    base = max(1.0, abs(x0f))

    # --- Initial evaluation at x0 ---
    fx0, ok0 = safe_eval(x0f)
    if ok0 and try_update(x0f, fx0):
        return best_x

    # --- Phase 1: Try multiple small steps to find a good partner point ---
    # For oscillatory functions, a small step is safer than 1e-4*base.
    # Try several step sizes and pick the one with smallest |f|.
    partner_x = None
    partner_fx = None
    partner_ok = False

    if ok0 and counter[0] < max_calls and _time_left() > 5.0:
        step_sizes = [1e-6 * base, 5e-6 * base, 1e-5 * base, 5e-5 * base, 1e-4 * base]
        for h in step_sizes:
            if counter[0] >= max_calls or _time_left() < 5.0:
                break
            x1 = x0f + h
            x2 = x0f - h
            fx1, ok1 = safe_eval(x1)
            if ok1 and try_update(x1, fx1):
                return best_x
            fx2, ok2 = safe_eval(x2)
            if ok2 and try_update(x2, fx2):
                return best_x

            # Track the best partner
            for xp, fxp, okp in [(x1, fx1, ok1), (x2, fx2, ok2)]:
                if okp and (partner_x is None or abs(fxp) < abs(partner_fx)):
                    partner_x = xp
                    partner_fx = fxp
                    partner_ok = True

            # If we found a very good point, stop expanding
            if best_abs < 1e-4:
                break

    # --- Phase 2: Secant method using best point and partner ---
    if best_abs > 1e-8 and counter[0] < max_calls and _time_left() > 5.0:
        x = float(best_x)
        fx, okx = safe_eval(x)
        if not okx:
            fx = fx0 if ok0 else float('nan')
            okx = ok0

        if okx and try_update(x, fx):
            return best_x

        # Use partner if available, else create one
        x1 = partner_x if partner_ok else x + 1e-5 * base
        fx1, ok1 = safe_eval(x1)
        if ok1 and try_update(x1, fx1):
            return best_x

        if okx and ok1:
            for _ in range(30):
                if counter[0] >= max_calls or _time_left() < 5.0:
                    break
                if abs(fx) <= 1e-8:
                    return x
                if abs(fx1) <= 1e-8:
                    return x1
                denom = fx1 - fx
                if abs(denom) < 1e-300:
                    break
                x_new = x1 - fx1 * (x1 - x) / denom
                if not math.isfinite(x_new):
                    break
                # Damp the step to avoid wild oscillations
                step = x_new - x1
                max_step = 1.0 * max(1.0, abs(x1))
                if abs(step) > max_step:
                    x_new = x1 + max_step * math.copysign(1.0, step)
                fx_new, ok_new = safe_eval(x_new)
                if not ok_new:
                    break
                if try_update(x_new, fx_new):
                    return best_x
                x, fx = x1, fx1
                x1, fx1 = x_new, fx_new

    # --- Phase 3: IQI from best point with two nearby partners ---
    if best_abs > 1e-8 and counter[0] < max_calls and _time_left() > 5.0:
        x_c = float(best_x)
        fc, okc = safe_eval(x_c)
        if not okc:
            fc = fx0 if ok0 else float('nan')
            okc = ok0

        h1 = 1e-5 * base
        h2 = 2e-5 * base
        x_a = x_c + h1
        x_b = x_c - h2
        fa, oka = safe_eval(x_a)
        if oka and try_update(x_a, fa):
            return best_x
        fb, okb = safe_eval(x_b)
        if okb and try_update(x_b, fb):
            return best_x

        if okc and oka and okb:
            pts = [(x_c, fc), (x_a, fa), (x_b, fb)]
            for _ in range(25):
                if counter[0] >= max_calls or _time_left() < 5.0:
                    break

                # Check if any is a root
                for xp, fp in pts:
                    if abs(fp) <= 1e-8:
                        return xp

                x_a, f_a = pts[0]
                x_b, f_b = pts[1]
                x_c, f_c = pts[2]

                # Inverse quadratic interpolation
                denom_a = (f_a - f_b) * (f_a - f_c)
                denom_b = (f_b - f_a) * (f_b - f_c)
                denom_c = (f_c - f_a) * (f_c - f_b)

                if abs(denom_a) < 1e-300 or abs(denom_b) < 1e-300 or abs(denom_c) < 1e-300:
                    # Fallback to secant
                    if abs(f_b - f_c) < 1e-300:
                        break
                    x_new = x_b - f_b * (x_b - x_c) / (f_b - f_c)
                else:
                    num_a = (0 - f_b) * (0 - f_c)
                    num_b = (0 - f_a) * (0 - f_c)
                    num_c = (0 - f_a) * (0 - f_b)
                    x_new = x_a * num_a / denom_a + x_b * num_b / denom_b + x_c * num_c / denom_c

                if not math.isfinite(x_new):
                    break

                # Damp the step
                ref_x = min(pts, key=lambda p: abs(p[0] - x_new))[0]
                step = x_new - ref_x
                max_step = 1.5 * max(1.0, abs(ref_x))
                if abs(step) > max_step:
                    x_new = ref_x + max_step * math.copysign(1.0, step)

                fx_new, ok_new = safe_eval(x_new)
                if not ok_new:
                    break
                if try_update(x_new, fx_new):
                    return best_x

                # Update points: keep the three with smallest |f|
                all_pts = pts + [(x_new, fx_new)]
                all_pts.sort(key=lambda p: abs(p[1]))
                pts = all_pts[:3]

    # --- Phase 4: Newton-like step using derivative estimate ---
    if best_abs > 1e-8 and counter[0] < max_calls and _time_left() > 5.0:
        x = float(best_x)
        fx, okx = safe_eval(x)
        if okx and try_update(x, fx):
            return best_x

        if okx:
            # Estimate derivative using central difference with small h
            h = 1e-7 * max(1.0, abs(x))
            x_plus = x + h
            x_minus = x - h
            fp, okp = safe_eval(x_plus)
            fm, okm = safe_eval(x_minus)
            if okp and okm:
                fp, okp = safe_eval(x_plus)
                fm, okm = safe_eval(x_minus)
                if okp and okm:
                    deriv = (fp - fm) / (2.0 * h)
                    if abs(deriv) > 1e-300:
                        for _ in range(15):
                            if counter[0] >= max_calls or _time_left() < 5.0:
                                break
                            if abs(fx) <= 1e-8:
                                return x
                            deriv = (fp - fm) / (2.0 * h)
                            if abs(deriv) < 1e-300:
                                break
                            x_new = x - fx / deriv
                            if not math.isfinite(x_new):
                                break
                            # Damp
                            step = x_new - x
                            max_step = 1.0 * max(1.0, abs(x))
                            if abs(step) > max_step:
                                x_new = x + max_step * math.copysign(1.0, step)
                            fx_new, ok_new = safe_eval(x_new)
                            if not ok_new:
                                break
                            if try_update(x_new, fx_new):
                                return best_x
                            x = x_new
                            fx = fx_new
                            # Re-estimate derivative
                            h = 1e-7 * max(1.0, abs(x))
                            x_plus = x + h
                            x_minus = x - h
                            fp, okp = safe_eval(x_plus)
                            fm, okm = safe_eval(x_minus)
                            if not (okp and okm):
                                break

    # --- Phase 5: Multi-scale dense sampling to find a bracket ---
    bracket_found = False
    a_br, b_br = None, None
    fa_br, fb_br = None, None

    scales = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0]
    n_samples = [7, 9, 11, 13, 15, 17, 19, 21, 25, 30, 35, 40, 45, 50, 55, 60, 65]

    for idx, scale in enumerate(scales):
        if counter[0] >= max_calls or _time_left() < 3.0:
            break
        h = scale * base
        n = n_samples[idx]
        half = n // 2
        points = []
        for i in range(-half, half + 1):
            p = x0f + i * h
            if p not in cache:
                val, ok = safe_eval(p)
                if ok and try_update(p, val):
                    return best_x
            points.append(p)

        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            v1, ok1 = cache[p1]
            v2, ok2 = cache[p2]
            if not ok1 or not ok2:
                continue
            if v1 == 0.0:
                return p1
            if v2 == 0.0:
                return p2
            if v1 * v2 < 0.0:
                a_br, b_br = p1, p2
                fa_br, fb_br = v1, v2
                bracket_found = True
                break
        if bracket_found:
            break

    # --- Phase 6: Brent's method on the bracket ---
    if bracket_found:
        try:
            res = _brentq(wf, a_br, b_br, xtol=1e-14, rtol=1e-14, maxiter=50)
            if math.isfinite(res):
                fr, okr = safe_eval(res)
                if okr and try_update(res, fr):
                    return best_x
                if okr and abs(fr) < best_abs:
                    best_x = res
                    best_abs = abs(fr)
        except Exception:
            pass

    # --- Phase 7: Final high-precision polish around best_x ---
    if best_abs > 1e-8 and counter[0] < max_calls and _time_left() > 2.0:
        # Fine local scan
        for scale in [1e-5, 1e-6, 1e-7, 1e-8]:
            if counter[0] >= max_calls or _time_left() < 2.0:
                break
            h = scale * max(1.0, abs(best_x))
            for i in range(-5, 6):
                if counter[0] >= max_calls:
                    break
                p = best_x + i * h
                if p not in cache:
                    val, ok = safe_eval(p)
                    if ok and try_update(p, val):
                        return best_x

        # Secant polish on best point
        x = float(best_x)
        fx, okx = safe_eval(x)
        if okx and try_update(x, fx):
            return best_x
        if okx:
            h = 1e-6 * max(1.0, abs(x))
            x1 = x + h
            fx1, ok1 = safe_eval(x1)
            if ok1 and try_update(x1, fx1):
                return best_x
            if okx and ok1:
                for _ in range(20):
                    if counter[0] >= max_calls or _time_left() < 2.0:
                        break
                    if abs(fx) <= 1e-8:
                        return x
                    if abs(fx1) <= 1e-8:
                        return x1
                    denom = fx1 - fx
                    if abs(denom) < 1e-300:
                        break
                    x_new = x1 - fx1 * (x1 - x) / denom
                    if not math.isfinite(x_new):
                        break
                    step = x_new - x1
                    max_step = 1.0 * max(1.0, abs(x1))
                    if abs(step) > max_step:
                        x_new = x1 + max_step * math.copysign(1.0, step)
                    fx_new, ok_new = safe_eval(x_new)
                    if not ok_new:
                        break
                    if try_update(x_new, fx_new):
                        return best_x
                    x, fx = x1, fx1
                    x1, fx1 = x_new, fx_new

    return best_x


def _brentq(f, a, b, xtol=1e-14, rtol=1e-14, maxiter=50):
    """Brent's method implementation."""
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have different signs")
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa
    c = a
    fc = fa
    d = b - a
    e = d
    for _ in range(maxiter):
        if abs(b - a) < xtol + rtol * abs(b):
            return b
        if fc * fb > 0:
            c = a
            fc = fa
            d = b - c
            e = d
        if abs(fc) < abs(fb):
            c = b
            fc = fb
            a = b
            fa = fb
            d = b - c
            e = d
        # Inverse quadratic interpolation
        tol = 2e-16 * max(1.0, abs(b))
        tm = 0.5 * (c - b)
        if abs(tm) <= tol or fc == fb:
            s = fc / (fc - fb)
            e = d
            d = tm * s
        else:
            r = fc / fa
            q = fb / fa
            s = fb / fc
            if c == a:
                d = tm * (2 * tm * (fb / (fc - fa) - 1) + s * (s - 1))
            else:
                d = tm * (2 * tm * (fb / (fc - fa) - 1) + s * (s - 1))
            if abs(d) >= abs(e):
                d = tm
                e = d
            else:
                e = d
        if d > 0:
            d = -d
        a = b
        fa = fb
        b += d
        fb = f(b)
    return b
