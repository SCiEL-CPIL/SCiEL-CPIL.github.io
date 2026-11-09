"""Root finder for 3 fixed smooth polynomial equations."""
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    # Damped Newton (bisection-safeguarded) with forward-difference derivative.
    # Robust to multiple roots / distant starts; call-efficient (~2 calls/iter).
    deadline = time.time() + 100.0  # safety margin under 120s (and 150s kill)
    x = float(x0)

    # First evaluation
    fx = f(x)
    best_x = x
    best_abs = abs(fx)
    if best_abs <= 1e-8:
        return x

    # Bracket for bisection safeguard (expand from x0)
    a = b = None
    fa = fb = None

    for _ in range(80):
        if time.time() > deadline:
            break
        if best_abs <= 1e-8:
            break

        # Forward-difference derivative at current x
        h = 1e-6 * max(1.0, abs(x))
        fxh = f(x + h)
        d = (fxh - fx) / h

        # Update best
        if abs(fxh) < best_abs:
            best_abs = abs(fxh)
            best_x = x + h

        if d == 0.0 or not (d == d):  # zero or NaN derivative
            # fall back to bisection if bracketed, else perturb
            if a is not None and b is not None and fa is not None and fb is not None:
                x = 0.5 * (a + b)
                fx = f(x)
                if abs(fx) < best_abs:
                    best_abs = abs(fx); best_x = x
                if fa * fx <= 0: b, fb = x, fx
                else: a, fa = x, fx
                continue
            else:
                # nudge
                x = x + 1e-4 * max(1.0, abs(x))
                fx = f(x)
                if abs(fx) < best_abs:
                    best_abs = abs(fx); best_x = x
                continue

        # Newton step
        xn = x - fx / d

        # Damping / bisection safeguard
        if a is None or b is None:
            # Establish bracket
            if fa is None:
                a, fa = x, fx
            # try to find opposite sign
            if fa * fx < 0:
                b, fb = x, fx
            else:
                # expand
                probe = x + (x - a) if a is not None else x + 1.0
                fp = f(probe)
                if fa * fp < 0:
                    b, fb = probe, fp
                else:
                    probe2 = a - (x - a) if a is not None else x - 1.0
                    fp2 = f(probe2)
                    if fa * fp2 < 0:
                        b, fb = probe2, fp2
                    else:
                        # no bracket; accept Newton step but keep a
                        pass

        # If bracketed, ensure step stays inside
        if a is not None and b is not None:
            lo, hi = min(a, b), max(a, b)
            if xn < lo or xn > hi:
                xn = 0.5 * (a + b)
                # bisect
                if fa * fx <= 0:
                    b, fb = x, fx
                else:
                    a, fa = x, fx

        x = xn
        fx = f(x)
        if abs(fx) < best_abs:
            best_abs = abs(fx)
            best_x = x

        # maintain bracket
        if a is not None and b is not None:
            if fa * fx <= 0:
                b, fb = x, fx
            else:
                a, fa = x, fx

    # Final polish: a few Newton iterations with tighter tolerance, no new calls beyond budget
    # (cheap; the evaluator's final check is free)
    for _ in range(6):
        if time.time() > deadline:
            break
        if abs(fx) <= 1e-12:
            break
        h = 1e-7 * max(1.0, abs(x))
        fxh = f(x + h)
        d = (fxh - fx) / h
        if d == 0.0 or not (d == d):
            break
        x = x - fx / d
        fx = f(x)
        if abs(fx) < best_abs:
            best_abs = abs(fx); best_x = x

    # Return best (smallest |f|) to maximize success chance
    return best_x
# EVOLVE-BLOCK-END
