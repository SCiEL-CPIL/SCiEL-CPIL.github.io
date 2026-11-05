"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Strategy: bracket + Illinois (guaranteed convergence for ill-conditioned f),
    with Brent-style safeguarded IQI polish, and derivative-free Newton/secant
    fallbacks when no bracket is found.
    """
    TOL = 1e-8
    MAX_EVALS = 500
    DEADLINE = time.monotonic() + 115.0  # safety margin before 120s kill

    evals = 0
    best_x = float(x0)
    best_f = None
    best_abs = float('inf')

    def call_f(x):
        """Safe evaluation: counts evals, guards deadline, tracks best."""
        nonlocal evals, best_x, best_f, best_abs
        if evals >= MAX_EVALS or time.monotonic() > DEADLINE:
            return None
        evals += 1
        try:
            val = f(x)
            val = float(val)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            return None
        if not math.isfinite(val):
            return None
        abs_val = abs(val)
        if abs_val < best_abs:
            best_abs = abs_val
            best_x = x
            best_f = val
        return val

    def is_good(x, fx):
        if fx is None:
            return False
        return abs(fx) <= TOL

    # ------------------------------------------------------------------
    # Illinois algorithm: guaranteed convergence via bracket + shrink
    # ------------------------------------------------------------------
    def illinois_solve(a, b, fa, fb, max_iter=120):
        """Illinois: regula falsi + shrink larger-|f| endpoint by half."""
        if fa is None or fb is None:
            return None
        if abs(fa) <= TOL:
            return a
        if abs(fb) <= TOL:
            return b
        if fa * fb > 0:
            return None
        try:
            lo, hi = a, b
            flo, fhi = fa, fb
            for _ in range(max_iter):
                if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 2:
                    break
                if abs(flo) <= TOL:
                    return lo
                if abs(fhi) <= TOL:
                    return hi
                if flo * fhi > 0:
                    break
                # Regula falsi step
                if abs(flo - fhi) > 1e-300:
                    xs = hi - fhi * (hi - lo) / (fhi - flo)
                else:
                    xs = (lo + hi) / 2.0
                if not (lo < xs < hi):
                    xs = (lo + hi) / 2.0
                fx = call_f(xs)
                if fx is None:
                    break
                if abs(fx) <= TOL:
                    return xs
                if flo * fx < 0:
                    hi, fhi = xs, fx
                else:
                    lo, flo = xs, fx
                # Illinois: shrink endpoint with larger |f| by half
                if abs(flo) > abs(fhi):
                    lo = (lo + hi) / 2.0
                    flo = call_f(lo)
                    if flo is None:
                        break
                else:
                    hi = (lo + hi) / 2.0
                    fhi = call_f(hi)
                    if fhi is None:
                        break
                if abs(hi - lo) < 1e-15 * max(1.0, abs(lo), abs(hi)):
                    break
            # Return better endpoint
            if abs(flo) <= abs(fhi):
                return lo
            else:
                return hi
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Brent-style safeguarded inverse quadratic interpolation
    # ------------------------------------------------------------------
    def brent_polish(a, b, fa, fb, max_iter=80):
        """
        Brent's method: IQR + bisection fallback.
        Requires fa*fb < 0.
        """
        if fa is None or fb is None or fa * fb > 0:
            return None
        if abs(fa) <= TOL:
            return a
        if abs(fb) <= TOL:
            return b
        try:
            # Invariant: |f(b)| <= |f(c)|, fa*fb < 0
            c = a
            fc = fa
            d = b - a
            e = d
            for _ in range(max_iter):
                if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 2:
                    break
                if abs(fb) <= TOL:
                    return b
                if fa * fb > 0:
                    a, fa = c, fc
                    d = b - a
                    e = d
                if abs(fc) < abs(fb):
                    a, fa = b, fb
                    b, fb = c, fc
                    c, fc = a, fa
                tol1 = 2 * 2.22e-16 * abs(b)
                delta = 0.5 * (c - a)
                if abs(delta) <= tol1 or fb == 0:
                    return b
                if abs(e) < tol1 or abs(fa) <= abs(fb):
                    # Bisection
                    e = delta
                    d = fb / fa
                else:
                    # Inverse quadratic interpolation
                    S = fa / fc
                    T1 = fb / fc
                    T2 = fb / fa
                    d = T1 * (T1 - S) * (b - c) / (fa - fc)
                    S = T1 * T2 * (c - a) / (fa - fc)
                    if (d > 0.5 * delta * (c - a)) or (d < 0.5 * delta * (b - c)):
                        d = delta
                    if abs(c - b) < 3 * tol1 / 2:
                        d = delta * ((c - b) if (fb >= 0) else (b - c)) / abs(fb)
                        if abs(d) < tol1:
                            d = delta
                e = d
                b = b + d
                fb = call_f(b)
                if fb is None:
                    break
            if abs(fb) <= TOL:
                return b
            return b
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Derivative-free Newton (two-point secant derivative)
    # ------------------------------------------------------------------
    def newton_raphson(x0, max_iter=30):
        """Newton with numerical derivative; damped to stay safe."""
        x = float(x0)
        fx = call_f(x)
        if fx is None:
            return x, None
        if is_good(x, fx):
            return x, fx
        for _ in range(max_iter):
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 2:
                break
            h = 1e-6 * max(1.0, abs(x))
            fx2 = call_f(x + h)
            if fx2 is None:
                break
            d = (fx2 - fx) / h
            if not math.isfinite(d) or abs(d) < 1e-15:
                break
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            # Damp large steps
            if abs(x_new - x) > 10 * max(1.0, abs(x)):
                x_new = x + math.copysign(10 * max(1.0, abs(x)), x_new - x)
            fx_new = call_f(x_new)
            if fx_new is None:
                break
            if is_good(x_new, fx_new):
                return x_new, fx_new
            if abs(fx_new) < abs(fx):
                x, fx = x_new, fx_new
            else:
                # Try half step
                x_half = x - 0.5 * (fx / d)
                fx_half = call_f(x_half)
                if fx_half is not None and abs(fx_half) < abs(fx):
                    x, fx = x_half, fx_half
                else:
                    break
        return x, fx

    # ------------------------------------------------------------------
    # Secant method
    # ------------------------------------------------------------------
    def secant(x0, x1, max_iter=40):
        """Secant: derivative-free, fast convergence near root."""
        x = float(x0)
        x_prev = float(x1)
        fx = call_f(x)
        fx_prev = call_f(x_prev)
        if fx is None and fx_prev is None:
            return x, None
        if fx is None:
            return x_prev, fx_prev
        if fx_prev is None:
            return x, fx
        if is_good(x, fx):
            return x, fx
        if is_good(x_prev, fx_prev):
            return x_prev, fx_prev
        for _ in range(max_iter):
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 2:
                break
            if fx == fx_prev:
                break
            x_new = x - fx * (x - x_prev) / (fx - fx_prev)
            if not math.isfinite(x_new):
                break
            fx_new = call_f(x_new)
            if fx_new is None:
                break
            if is_good(x_new, fx_new):
                return x_new, fx_new
            x_prev, fx_prev = x, fx
            x, fx = x_new, fx_new
        return x, fx

    # ------------------------------------------------------------------
    # Quadratic model prediction
    # ------------------------------------------------------------------
    def model_predict(points):
        """Fit quadratic to last 3 points, predict zero crossing."""
        if len(points) < 2:
            return None
        if len(points) < 3:
            (x1, y1), (x2, y2) = points[-2:]
            if abs(y1 - y2) < 1e-300:
                return None
            xr = x1 - y1 * (x2 - x1) / (y2 - y1)
            if not math.isfinite(xr):
                return None
            return xr
        (x1, y1), (x2, y2), (x3, y3) = points[-3:]
        if abs(x1 - x2) < 1e-15 or abs(x2 - x3) < 1e-15 or abs(x1 - x3) < 1e-15:
            return None
        try:
            A = np.array([[x1**2, x1, 1],
                          [x2**2, x2, 1],
                          [x3**2, x3, 1]], dtype=float)
            b_vec = np.array([y1, y2, y3], dtype=float)
            coeffs = np.linalg.solve(A, b_vec)
            a, b, c = coeffs
        except np.linalg.LinAlgError:
            return None
        if abs(a) < 1e-300:
            if abs(b) < 1e-300:
                return None
            xr = -c / b
        else:
            disc = b * b - 4 * a * c
            if disc < -1e-12:
                return None
            disc = max(0, disc)
            sqrt_disc = np.sqrt(disc)
            r1 = (-b + sqrt_disc) / (2 * a)
            r2 = (-b - sqrt_disc) / (2 * a)
            xmid = (x1 + x2 + x3) / 3.0
            xr = r1 if abs(r1 - xmid) < abs(r2 - xmid) else r2
        if not math.isfinite(xr):
            return None
        xref = best_x
        if abs(xr - xref) > 100 * max(1.0, abs(xref)):
            return None
        return xr

    # ==================================================================
    # Main search logic
    # ==================================================================
    x0 = float(x0)
    fx0 = call_f(x0)
    if fx0 is not None and is_good(x0, fx0):
        return x0

    points = [(x0, fx0)] if fx0 is not None else []

    # --- Phase 1: Local expansion to find a bracket ---
    scales = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    bracket_found = False
    for s in scales:
        if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 10:
            break
        a = x0 - s
        b = x0 + s
        fa = call_f(a)
        fb = call_f(b)
        if fa is not None:
            points.append((a, fa))
        if fb is not None:
            points.append((b, fb))
        if fa is not None and is_good(a, fa):
            return a
        if fb is not None and is_good(b, fb):
            return b

        # Check sign change
        if fa is not None and fx0 is not None and fa * fx0 < 0:
            root = illinois_solve(a, x0, fa, fx0)
            if root is not None:
                fr = call_f(root)
                if fr is not None and is_good(root, fr):
                    return root
            bracket_found = True
            break
        if fb is not None and fx0 is not None and fb * fx0 < 0:
            root = illinois_solve(x0, b, fx0, fb)
            if root is not None:
                fr = call_f(root)
                if fr is not None and is_good(root, fr):
                    return root
            bracket_found = True
            break
        if fa is not None and fb is not None and fa * fb < 0:
            root = illinois_solve(a, b, fa, fb)
            if root is not None:
                fr = call_f(root)
                if fr is not None and is_good(root, fr):
                    return root
            bracket_found = True
            break

    # --- Phase 2: If bracket found, polish with Brent ---
    if bracket_found:
        # Find the best bracket in points
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if points[i][1] * points[j][1] < 0:
                    a = min(points[i][0], points[j][0])
                    b = max(points[i][0], points[j][0])
                    fa = points[i][1] if points[i][0] == a else points[j][1]
                    fb = points[j][1] if points[j][0] == b else fa
                    root = brent_polish(a, b, fa, fb, max_iter=80)
                    if root is not None:
                        fr = call_f(root)
                        if fr is not None and is_good(root, fr):
                            return root
                        # If not good enough, try Illinois for guaranteed convergence
                        root2 = illinois_solve(a, b, fa, fb, max_iter=120)
                        if root2 is not None:
                            fr2 = call_f(root2)
                            if fr2 is not None and is_good(root2, fr2):
                                return root2
                            if fr2 is not None and abs(fr2) < best_abs:
                                best_x = root2
                                best_abs = abs(fr2)
                    break
            if bracket_found:
                break

    # --- Phase 3: Derivative-free fallbacks (no bracket or not converged) ---
    # Newton
    if time.monotonic() < DEADLINE and evals < MAX_EVALS - 10:
        x_res, f_res = newton_raphson(best_x)
        if f_res is not None and is_good(x_res, f_res):
            return x_res

    # Secant with nearby points
    if time.monotonic() < DEADLINE and evals < MAX_EVALS - 10:
        h = 1e-4 * max(1.0, abs(best_x))
        x_res, f_res = secant(best_x, best_x + h)
        if f_res is not None and is_good(x_res, f_res):
            return x_res

    # Secant with wider interval
    if time.monotonic() < DEADLINE and evals < MAX_EVALS - 10:
        h = 1.0
        x_res, f_res = secant(best_x - h, best_x + h)
        if f_res is not None and is_good(x_res, f_res):
            return x_res

    # --- Phase 4: Model-guided local search ---
    for iteration in range(30):
        if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 10:
            break
        x_pred = model_predict(points)
        if x_pred is None:
            xbase = best_x
            x_pred = xbase + 0.1 * max(1.0, abs(xbase)) * (
                1.0 if iteration % 2 == 0 else -1.0
            )
        if evals >= MAX_EVALS - 5:
            break
        fx = call_f(x_pred)
        if fx is None:
            break
        if is_good(x_pred, fx):
            return x_pred
        points.append((x_pred, fx))

        # Check for new brackets
        for i in range(len(points) - 1, 0, -1):
            xi, fi = points[i]
            xj, fj = points[i - 1]
            if fi * fj < 0:
                root = illinois_solve(xj, xi, fj, fi)
                if root is not None:
                    fr = call_f(root)
                    if fr is not None and is_good(root, fr):
                        return root
                break

        if len(points) > 20:
            points = points[-10:]

    # --- Phase 5: Broader multi-scale exploration ---
    for radius in [0.5, 1, 2, 5, 10, 50, 100, 500]:
        if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 20:
            break
        for _ in range(3):
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 10:
                break
            x_try = x0 + radius * (1.0 if _ % 2 == 0 else -1.0)
            a = x_try - radius * 0.1
            b = x_try + radius * 0.1
            if evals >= MAX_EVALS - 5:
                break
            fa = call_f(a)
            fb = call_f(b)
            if fa is not None:
                points.append((a, fa))
            if fb is not None:
                points.append((b, fb))
            if fa is not None and is_good(a, fa):
                return a
            if fb is not None and is_good(b, fb):
                return b
            if fa is not None and fb is not None and fa * fb < 0:
                root = illinois_solve(a, b, fa, fb)
                if root is not None:
                    fr = call_f(root)
                    if fr is not None and is_good(root, fr):
                        return root
                break

    # --- Phase 6: Final polish on any bracket ---
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if points[i][1] * points[j][1] < 0:
                a = min(points[i][0], points[j][0])
                b = max(points[i][0], points[j][0])
                fa = points[i][1] if points[i][0] == a else points[j][1]
                fb = points[j][1] if points[j][0] == b else fa
                root = brent_polish(a, b, fa, fb, max_iter=60)
                if root is not None:
                    fr = call_f(root)
                    if fr is not None and is_good(root, fr):
                        return root
                    root2 = illinois_solve(a, b, fa, fb, max_iter=100)
                    if root2 is not None:
                        fr2 = call_f(root2)
                        if fr2 is not None and is_good(root2, fr2):
                            return root2
                        if fr2 is not None and abs(fr2) < best_abs:
                            best_x = root2
                            best_abs = abs(fr2)

    return best_x
# EVOLVE-BLOCK-END
