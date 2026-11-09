"""Direct approach: efficient bracketing + safeguarded Newton/bisection with polish."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    x0 = float(x0)
    T0 = time.time()
    DEADLINE = 145.0          # safety margin before 150 s hard kill
    TOL = 1e-12               # aim well below 1e-8
    HARD_TOL = 1e-8           # contract tolerance
    MAX_REFINE_ITERS = 300

    def elapsed_ok():
        return (time.time() - T0) < DEADLINE

    # Evaluate start
    try:
        f0 = f(x0)
    except Exception:
        f0 = float('nan')

    best_x = x0
    best_val = abs(f0) if (f0 == f0 and math.isfinite(f0)) else float('inf')

    def track(x, fx):
        nonlocal best_x, best_val
        if fx == fx and math.isfinite(fx):
            av = abs(fx)
            if av < best_val:
                best_val = av
                best_x = x

    track(x0, f0)

    # If start is already good
    if f0 == f0 and math.isfinite(f0) and abs(f0) <= HARD_TOL:
        return x0

    # ------------------------------------------------------------------
    # 1) Find a bracket [a, b] with a sign change, searching outward
    #    with adaptive geometric steps and dense local sampling.
    # ------------------------------------------------------------------
    a = b = None
    fa = fb = None

    base = max(1.0, abs(x0))

    # Phase 1: Coarse outward scan with geometric step growth
    # Try both directions, growing step geometrically
    step = base * 1e-3
    lo = x0
    flo = f0
    found = False

    # Scan in positive direction
    for i in range(60):
        if not elapsed_ok():
            break
        hi = lo + step
        try:
            fhi = f(hi)
        except Exception:
            fhi = float('nan')
        track(hi, fhi)
        if fhi == fhi and math.isfinite(fhi) and flo == flo and math.isfinite(flo) and flo * fhi < 0:
            a, b, fa, fb = lo, hi, flo, fhi
            found = True
            break
        lo = hi
        if fhi == fhi and math.isfinite(fhi):
            flo = fhi
        else:
            flo = float('nan')
        # Grow step geometrically
        if i < 5:
            step = step * 1.5
        else:
            step = step * 2.0

    if not found:
        # Scan in negative direction
        step = base * 1e-3
        hi = x0
        fhi = f0
        for i in range(60):
            if not elapsed_ok():
                break
            lo = hi - step
            try:
                flo = f(lo)
            except Exception:
                flo = float('nan')
            track(lo, flo)
            if flo == flo and math.isfinite(flo) and fhi == fhi and math.isfinite(fhi) and flo * fhi < 0:
                a, b, fa, fb = lo, hi, flo, fhi
                found = True
                break
            hi = lo
            if flo == flo and math.isfinite(flo):
                fhi = flo
            else:
                fhi = float('nan')
            if i < 5:
                step = step * 1.5
            else:
                step = step * 2.0

    # Phase 2: If no bracket found, do denser local scans at multiple radii
    if not found:
        radii = [base * 1e-3, base * 1e-2, base * 1e-1, base, base * 10, base * 100]
        for r in radii:
            if not elapsed_ok():
                break
            # Adaptive number of samples based on radius
            n = int(min(2000, max(20, 40 * math.sqrt(r / max(base, 1e-12)))))
            left = x0 - r
            right = x0 + r
            prev_x = left
            try:
                prev_f = f(prev_x)
            except Exception:
                prev_f = float('nan')
            track(prev_x, prev_f)
            for i in range(1, n + 1):
                if not elapsed_ok():
                    break
                x_i = left + (right - left) * i / n
                try:
                    f_i = f(x_i)
                except Exception:
                    f_i = float('nan')
                track(x_i, f_i)
                if f_i == f_i and math.isfinite(f_i) and prev_f == prev_f and math.isfinite(prev_f) and prev_f * f_i < 0:
                    a, b, fa, fb = prev_x, x_i, prev_f, f_i
                    found = True
                    break
                prev_x, prev_f = x_i, f_i
            if found:
                break

    # ------------------------------------------------------------------
    # 2) Refine the bracket with safeguarded Newton + bisection.
    # ------------------------------------------------------------------
    if a is not None:
        # Ensure fa, fb correspond to a, b
        if fa is None or not math.isfinite(fa) or fa != fa:
            try:
                fa = f(a)
            except Exception:
                fa = float('nan')
        if fb is None or not math.isfinite(fb) or fb != fb:
            try:
                fb = f(b)
            except Exception:
                fb = float('nan')

        # Pick the side with smaller |f| as initial guess
        if abs(fa) <= abs(fb):
            x, fx = a, fa
            x2, f2 = b, fb
        else:
            x, fx = b, fb
            x2, f2 = a, fa

        for _ in range(MAX_REFINE_ITERS):
            if not elapsed_ok():
                break
            if fx == fx and math.isfinite(fx) and abs(fx) <= TOL:
                return x
            # update best
            track(x, fx)

            # Bisection midpoint as fallback
            mid = 0.5 * (x + x2)

            # Newton step (secant-based since we don't have derivative)
            used_newton = False
            if fx == fx and f2 == f2 and math.isfinite(fx) and math.isfinite(f2) and f2 != fx:
                denom = (f2 - fx) / (x2 - x) if x2 != x else 0.0
                if denom != 0 and math.isfinite(denom):
                    xn = x - fx / denom
                    # Accept Newton step only if it stays inside bracket
                    if (min(a, b) < xn < max(a, b)):
                        try:
                            fn = f(xn)
                        except Exception:
                            fn = float('nan')
                        track(xn, fn)
                        if fn == fn and math.isfinite(fn) and abs(fn) <= TOL:
                            return xn
                        if fn == fn and math.isfinite(fn):
                            # Update bracket on the side that still straddles
                            if fx * fn < 0:
                                x2, f2 = x, fx
                                x, fx = xn, fn
                            else:
                                x, fx = xn, fn
                            used_newton = True

            if not used_newton:
                # Bisection step
                try:
                    fm = f(mid)
                except Exception:
                    fm = float('nan')
                track(mid, fm)
                if fm == fm and math.isfinite(fm) and abs(fm) <= TOL:
                    return mid
                if fm == fm and math.isfinite(fm):
                    if fx == fx and math.isfinite(fx) and fx * fm < 0:
                        x2, f2 = mid, fm
                    else:
                        x, fx = mid, fm
                # If fm is NaN or inf, shrink toward side with finite value
                else:
                    if fx == fx and math.isfinite(fx):
                        x2, f2 = mid, float('nan')
                    else:
                        x, fx = mid, float('nan')

        # After refinement, return the best-seen point
        if best_val <= HARD_TOL:
            return best_x

    # ------------------------------------------------------------------
    # 3) No bracket found: do a short local secant/Newton-style attempt
    #    from x0 as a last resort, then return the best finite candidate.
    # ------------------------------------------------------------------
    try:
        x = x0
        fx = f0
        if fx != fx or not math.isfinite(fx):
            # Try to get a finite starting point
            for dx in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]:
                try:
                    f_try = f(x0 + dx)
                except Exception:
                    continue
                if f_try == f_try and math.isfinite(f_try):
                    x = x0 + dx
                    fx = f_try
                    track(x, fx)
                    break
                try:
                    f_try = f(x0 - dx)
                except Exception:
                    continue
                if f_try == f_try and math.isfinite(f_try):
                    x = x0 - dx
                    fx = f_try
                    track(x, fx)
                    break

        if fx != fx or not math.isfinite(fx):
            # Can't even get a finite value, return best
            if best_x == best_x and math.isfinite(best_val):
                return best_x
            return x0

        for _ in range(80):
            if not elapsed_ok():
                break
            if fx == fx and math.isfinite(fx) and abs(fx) <= TOL:
                return x
            h = 1e-7 * max(1.0, abs(x))
            try:
                fx2 = f(x + h)
            except Exception:
                break
            if fx2 != fx2 or not math.isfinite(fx2):
                break
            d = (fx2 - fx) / h
            if d == 0 or d != d or not math.isfinite(d):
                break
            xn = x - fx / d
            if xn != xn or not math.isfinite(xn):
                break
            try:
                fn = f(xn)
            except Exception:
                break
            track(xn, fn)
            if fn == fn and math.isfinite(fn) and abs(fn) <= TOL:
                return xn
            # Damp if it overshoots wildly
            if abs(xn - x) > 1e6 * max(1.0, abs(x)):
                xn = 0.5 * (x + xn)
            x, fx = xn, fn
    except Exception:
        pass

    # Return best finite candidate seen
    if best_x == best_x and math.isfinite(best_val):
        return best_x
    return x0
# EVOLVE-BLOCK-END
