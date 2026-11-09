"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    # Robust safeguarded Newton + bisection with a wall-clock deadline and
    # call budget.  Targets the failure mode of the plain Newton-FD program
    # (a non-converging equation) while keeping mean_calls tiny on the easy
    # smooth-polynomial cases.
    DEADLINE = time.time() + 115.0   # safety margin under the 120s limit
    BUDGET = 470                     # headroom under the 500-call failure line

    call_count = [0]
    cache = {}

    def feval(x):
        # Cache by exact float value so repeated points are not double-counted.
        key = x
        if key in cache:
            return cache[key]
        if call_count[0] >= BUDGET or time.time() >= DEADLINE:
            # Return a sentinel; caller treats as "stop searching".
            return None
        call_count[0] += 1
        try:
            v = float(f(key))
        except Exception:
            v = float('nan')
        if not math.isfinite(v):
            v = float('nan')
        cache[key] = v
        return v

    x = float(x0)

    # Track the best x seen (smallest |f|) as a safety net.
    best_x = x
    best_fx = feval(x)
    if best_fx is None:
        return x
    best_abs = abs(best_fx) if math.isfinite(best_fx) else float('inf')

    def record(xv, fxv):
        nonlocal best_x, best_abs
        if fxv is not None and math.isfinite(fxv):
            if abs(fxv) < best_abs:
                best_abs = abs(fxv)
                best_x = xv
                if abs(fxv) <= 1e-12:
                    return True
        return False

    # Initial check.
    if best_fx is not None and math.isfinite(best_fx) and abs(best_fx) <= 1e-8:
        return x

    # Try to establish a bracket [lo, hi] around x0 by expanding in both
    # directions.  A sign change guarantees bisection convergence.
    lo = hi = None
    fx0 = best_fx
    if math.isfinite(fx0) and fx0 != 0.0:
        step = max(1e-3, abs(x) * 1e-3)
        # Expand outward up to a few doublings.
        for _ in range(60):
            if time.time() >= DEADLINE or call_count[0] >= BUDGET:
                break
            xl = x - step
            xr = x + step
            fxl = feval(xl)
            fxr = feval(xr)
            if fxl is None or fxr is None:
                break
            if math.isfinite(fxl) and math.isfinite(fxr):
                if fxl == 0.0:
                    return xl
                if fxr == 0.0:
                    return xr
                if fx0 * fxl < 0.0:
                    lo, hi = xl, x
                    break
                if fx0 * fxr < 0.0:
                    lo, hi = x, xr
                    break
            step *= 2.0

    # Safeguarded Newton iteration.
    # We maintain a working interval [a, b] if a bracket exists; otherwise we
    # run pure Newton with a step-damping guard.
    if lo is not None and hi is not None:
        a, b = lo, hi
        fa = feval(a)
        fb = feval(b)
        # Ensure fa*fb <= 0 (should hold by construction).
        # Start Newton from x0 if it's inside the bracket, else from midpoint.
        if a <= x <= b:
            cx = x
            fcx = feval(cx)
        else:
            cx = 0.5 * (a + b)
            fcx = feval(cx)
        record(cx, fcx)
        if fcx is not None and math.isfinite(fcx) and abs(fcx) <= 1e-8:
            return cx

        for _ in range(200):
            if time.time() >= DEADLINE or call_count[0] >= BUDGET:
                break
            if fcx is None or not math.isfinite(fcx):
                # Derivative estimate failed; just bisect.
                mid = 0.5 * (a + b)
                fm = feval(mid)
                record(mid, fm)
                if fm is None:
                    break
                if not math.isfinite(fm):
                    break
                if abs(fm) <= 1e-8:
                    return mid
                if fa * fm <= 0.0:
                    b, fb = mid, fm
                else:
                    a, fa = mid, fm
                continue

            # Finite-difference derivative.
            h = 1e-6 * max(1.0, abs(cx))
            f1 = feval(cx - h)
            f2 = feval(cx + h)
            if f1 is None or f2 is None:
                break
            if not (math.isfinite(f1) and math.isfinite(f2)):
                # Fall back to bisection.
                mid = 0.5 * (a + b)
                fm = feval(mid)
                record(mid, fm)
                if fm is None or not math.isfinite(fm):
                    break
                if abs(fm) <= 1e-8:
                    return mid
                if fa * fm <= 0.0:
                    b, fb = mid, fm
                else:
                    a, fa = mid, fm
                continue

            d = (f2 - f1) / (2.0 * h)
            if d == 0.0 or not math.isfinite(d):
                # Bisect.
                mid = 0.5 * (a + b)
                fm = feval(mid)
                record(mid, fm)
                if fm is None or not math.isfinite(fm):
                    break
                if abs(fm) <= 1e-8:
                    return mid
                if fa * fm <= 0.0:
                    b, fb = mid, fm
                else:
                    a, fa = mid, fm
                continue

            # Newton candidate.
            xn = cx - fcx / d
            # Clamp into bracket for safety (safeguarded Newton).
            if not (a < xn < b):
                xn = 0.5 * (a + b)
            fxn = feval(xn)
            record(xn, fxn)
            if fxn is None or not math.isfinite(fxn):
                break
            if abs(fxn) <= 1e-8:
                return xn
            if fa * fxn <= 0.0:
                b, fb = xn, fxn
            else:
                a, fa = xn, fxn
            cx, fcx = xn, fxn
            # Convergence by interval width.
            if abs(b - a) <= 1e-12 * max(1.0, abs(cx)):
                break
    else:
        # No bracket found: run damped Newton with a fallback to a local
        # search for a sign change.
        for _ in range(200):
            if time.time() >= DEADLINE or call_count[0] >= BUDGET:
                break
            fx = feval(x)
            if fx is None:
                break
            if not math.isfinite(fx):
                break
            record(x, fx)
            if abs(fx) <= 1e-8:
                return x
            h = 1e-6 * max(1.0, abs(x))
            f1 = feval(x - h)
            f2 = feval(x + h)
            if f1 is None or f2 is None:
                break
            if not (math.isfinite(f1) and math.isfinite(f2)):
                break
            d = (f2 - f1) / (2.0 * h)
            if d == 0.0 or not math.isfinite(d):
                break
            step = fx / d
            # Damp the step to avoid blow-ups.
            max_step = max(1.0, abs(x)) * 0.5
            if abs(step) > max_step:
                step = max_step if step > 0 else -max_step
            xn = x + step
            fxn = feval(xn)
            record(xn, fxn)
            if fxn is None or not math.isfinite(fxn):
                # Shrink step.
                x = x + step * 0.5
                continue
            if abs(fxn) <= 1e-8:
                return xn
            # If the step reduced |f|, accept; otherwise shrink.
            if abs(fxn) < abs(fx):
                x = xn
            else:
                # Try to find a sign change around x to enable bisection.
                step2 = max(1e-3, abs(x) * 1e-3)
                found = False
                for _ in range(30):
                    if time.time() >= DEADLINE or call_count[0] >= BUDGET:
                        break
                    xl = x - step2
                    xr = x + step2
                    fxl = feval(xl)
                    fxr = feval(xr)
                    if fxl is None or fxr is None:
                        break
                    if math.isfinite(fxl) and math.isfinite(fxr):
                        if fxl == 0.0:
                            return xl
                        if fxr == 0.0:
                            return xr
                        if fx * fxl < 0.0:
                            a, b, fa, fb = xl, x, fxl, fx
                            x = 0.5 * (a + b)
                            found = True
                            break
                        if fx * fxr < 0.0:
                            a, b, fa, fb = x, xr, fx, fxr
                            x = 0.5 * (a + b)
                            found = True
                            break
                    step2 *= 2.0
                if found:
                    # Continue with bisection from here.
                    for _ in range(120):
                        if time.time() >= DEADLINE or call_count[0] >= BUDGET:
                            break
                        mid = 0.5 * (a + b)
                        fm = feval(mid)
                        record(mid, fm)
                        if fm is None or not math.isfinite(fm):
                            break
                        if abs(fm) <= 1e-8:
                            return mid
                        if fa * fm <= 0.0:
                            b, fb = mid, fm
                        else:
                            a, fa = mid, fm
                        if abs(b - a) <= 1e-12 * max(1.0, abs(mid)):
                            break
                    break
                else:
                    x = x + step * 0.5

    # Final high-precision polish on the best candidate if we have budget.
    if call_count[0] < BUDGET - 5 and time.time() < DEADLINE - 1.0:
        px = best_x
        for _ in range(12):
            if time.time() >= DEADLINE or call_count[0] >= BUDGET:
                break
            fx = feval(px)
            if fx is None or not math.isfinite(fx):
                break
            if abs(fx) <= 1e-12:
                return px
            h = 1e-7 * max(1.0, abs(px))
            f1 = feval(px - h)
            f2 = feval(px + h)
            if f1 is None or f2 is None:
                break
            if not (math.isfinite(f1) and math.isfinite(f2)):
                break
            d = (f2 - f1) / (2.0 * h)
            if d == 0.0 or not math.isfinite(d):
                break
            step = fx / d
            max_step = max(1.0, abs(px)) * 0.25
            if abs(step) > max_step:
                step = max_step if step > 0 else -max_step
            pn = px + step
            fpn = feval(pn)
            if fpn is None or not math.isfinite(fpn):
                break
            if abs(fpn) < abs(fx):
                px = pn
                best_x = pn
                best_abs = abs(fpn)
            else:
                break

    # Return the best finite candidate.
    if math.isfinite(best_x):
        return best_x
    return x
# EVOLVE-BLOCK-END
