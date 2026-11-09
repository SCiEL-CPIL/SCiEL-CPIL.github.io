"""Root finder for smooth polynomial equations with call-budget awareness.

Strategy:
  1. Evaluate f(x0) and a small neighbourhood to get 3-5 points.
  2. If a sign change is found, use a bracketing method (Illinois /
     safeguarded secant) to converge to 1e-8 accuracy.
  3. If no sign change is found immediately, use a few secant/Newton
     iterations with damping to try to converge directly.
  4. As a fallback, do a wider bracketing search and then bracket.

All function calls are counted; we keep a running best and enforce a
call budget of 480 to stay well under the 500-call failure threshold.
"""
import time
import math

_DEADLINE_START = time.time()
_WALL_LIMIT = 145.0  # safety margin before 150s kill
_SAFETY_MARGIN = 5.0
_CALL_BUDGET = 480


def _time_left():
    return _WALL_LIMIT - _SAFETY_MARGIN - (time.time() - _DEADLINE_START)


def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def find_root(f, x0):
    if not _is_finite(x0):
        x0 = 0.0
    x = float(x0)

    best_x = x
    best_f = f(x)
    call_count = 1

    def safe_eval(x_val):
        nonlocal call_count, best_x, best_f
        if not _is_finite(x_val):
            return None
        fx_val = f(x_val)
        call_count += 1
        if not _is_finite(fx_val):
            return None
        if abs(fx_val) < abs(best_f):
            best_f = fx_val
            best_x = x_val
        return fx_val

    if abs(best_f) <= 1e-8:
        return best_x

    # ---- Phase 1: get a few nearby points for interpolation / bracketing ----
    # We'll collect up to 5 points in a small window around x0.
    h0 = 1e-4 * max(1.0, abs(x))
    # Points to try (relative offsets)
    offsets = [h0, -h0, 2.0 * h0, -2.0 * h0, 5.0 * h0, -5.0 * h0]
    pts = [(x, best_f)]
    for off in offsets:
        if call_count >= _CALL_BUDGET or _time_left() <= 0:
            break
        xi = x + off
        fi = safe_eval(xi)
        if fi is None:
            continue
        if abs(fi) <= 1e-8:
            return xi
        pts.append((xi, fi))

    # ---- Phase 2: try to find a sign-change bracket among the points ----
    bracket = None
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            (xa, fa), (xb, fb) = pts[i], pts[j]
            if fa * fb < 0:
                bracket = (min(xa, xb), max(xa, xb), fa, fb)
                break
        if bracket:
            break

    if bracket:
        # ---- Phase 3: Illinois / safeguarded-secant on the bracket ----
        a, b, fa, fb = bracket
        # Ensure |fa| <= |fb| for Illinois damping convention
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        for _ in range(80):
            if call_count >= _CALL_BUDGET or _time_left() <= 0:
                break
            if abs(fb - fa) < 1e-15:
                m = (a + b) / 2.0
            else:
                m = b - fb * (b - a) / (fb - fa)
            if not (a < m < b):
                m = (a + b) / 2.0
            fm = safe_eval(m)
            if fm is None:
                break
            if abs(fm) <= 1e-8:
                return m
            if fa * fm < 0:
                b, fb = m, fm
                fa *= 0.5
            else:
                a, fa = m, fm
                fb *= 0.5
        if abs(best_f) <= 1e-8:
            return best_x
        return best_x

    # ---- Phase 4: no local sign change; try secant from two nearest points ----
    # Pick the two points with the smallest |f| values (excluding x0 if possible)
    sorted_pts = sorted(pts, key=lambda p: abs(p[1]))
    (x1, f1), (x2, f2) = sorted_pts[0], sorted_pts[1]
    for _ in range(40):
        if call_count >= _CALL_BUDGET or _time_left() <= 0:
            break
        if abs(f2 - f1) < 1e-15:
            # derivative ~0; perturb
            h = 1e-3 * max(1.0, abs(x2))
            x3 = x2 + h
            f3 = safe_eval(x3)
            if f3 is None:
                break
            x1, f1 = x2, f2
            x2, f2 = x3, f3
            continue
        x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
        if not _is_finite(x_new):
            break
        # Damp large steps
        if abs(x_new - x2) > 10.0 * max(1.0, abs(x2)):
            x_new = x2 + 0.5 * (x_new - x2)
        f_new = safe_eval(x_new)
        if f_new is None:
            break
        if abs(f_new) <= 1e-8:
            return x_new
        if abs(f_new - f2) < 1e-14 and abs(x_new - x2) < 1e-12:
            break
        x1, f1 = x2, f2
        x2, f2 = x_new, f_new

    # ---- Phase 5: wider bracketing search ----
    if abs(best_f) > 1e-8:
        base = max(1.0, abs(x0))
        step = base * 0.5
        for i in range(1, 30):
            if call_count >= _CALL_BUDGET or _time_left() <= 0:
                break
            xa = x0 - i * step
            xb = x0 + i * step
            fa = safe_eval(xa)
            if fa is None:
                continue
            if abs(fa) <= 1e-8:
                return xa
            fb = safe_eval(xb)
            if fb is None:
                continue
            if abs(fb) <= 1e-8:
                return xb
            if fa * fb < 0:
                a, b, fa, fb = min(xa, xb), max(xa, xb), fa, fb
                if abs(fa) > abs(fb):
                    a, b = b, a
                    fa, fb = fb, fa
                for _ in range(60):
                    if call_count >= _CALL_BUDGET or _time_left() <= 0:
                        break
                    if abs(fb - fa) < 1e-15:
                        m = (a + b) / 2.0
                    else:
                        m = b - fb * (b - a) / (fb - fa)
                    if not (a < m < b):
                        m = (a + b) / 2.0
                    fm = safe_eval(m)
                    if fm is None:
                        break
                    if abs(fm) <= 1e-8:
                        return m
                    if fa * fm < 0:
                        b, fb = m, fm
                        fa *= 0.5
                    else:
                        a, fa = m, fm
                        fb *= 0.5
                break

    if abs(best_f) <= 1e-8:
        return best_x
    return best_x


# EVOLVE-BLOCK-END
