"""Root finding for ill-conditioned equations: pure global bracket-first strategy.

Design family:
  1. Do NOT use Newton or finite-difference derivatives (which can diverge or
     stagnate on ill-conditioned functions and waste evaluations).
  2. Aggressively search for a bracket [a, b] where f(a) and f(b) have opposite
     signs using two complementary mechanisms:
       a. Geometric expansion: sample points x0 + k*delta and x0 - k*delta for
          increasing k, looking for a sign change with the start point or other
          samples.
       b. Golden-section minimization: minimize |f(x)| over an expanding window.
          If the minimum |f| is very small, a root is nearby. We then do a local
          sign-change search around that minimum.
  3. Once a bracket is found, use bisection (not Brent) to converge to the root
     with high precision. Bisection is robust, monotonic, and requires no
     derivatives.
  4. Track the global minimum |f(x)| seen across all evaluations. Return the x
     with the smallest |f(x)| if we run out of budget before finding a sign-change
     bracket, maximizing partial credit.
"""

import math
import time

# Hard budgets
EVAL_CAP = 500
TIME_LIMIT = 135.0  # Safety margin below 150s kill
TOL = 1e-8
TOL_TIGHT = 1e-12


class _EvalBudget:
    __slots__ = ("f", "count", "deadline", "cap")

    def __init__(self, f, cap=EVAL_CAP, time_limit=TIME_LIMIT):
        self.f = f
        self.cap = cap
        self.count = 0
        self.deadline = time.time() + time_limit

    def spent(self):
        return self.count >= self.cap

    def expired(self):
        return time.time() >= self.deadline

    def __call__(self, x):
        if self.spent() or self.expired():
            raise _BudgetExhausted()
        self.count += 1
        try:
            y = self.f(x)
        except Exception:
            y = math.nan
        try:
            y = float(y)
        except Exception:
            y = math.nan
        return y


class _BudgetExhausted(Exception):
    pass


def _safe_eval(call, x):
    """Return (value, ok) without raising on budget exhaustion."""
    try:
        y = call(x)
    except _BudgetExhausted:
        return math.inf, False
    ok = math.isfinite(y)
    return (y if ok else math.inf), ok


def _golden_section_minimize(call, a, b, best, max_iter=30):
    """
    Minimize |f(x)| on [a, b] using golden-section search.
    Updates best = [x_min, f_abs_min] with the best point found.
    Returns (x_min, f_abs_min) of the best point found during the search.
    """
    phi = (math.sqrt(5) - 1) / 2  # golden ratio conjugate ~0.618
    c = b - phi * (b - a)
    d = a + phi * (b - a)

    fc, _ = _safe_eval(call, c)
    fd, _ = _safe_eval(call, d)

    if math.isfinite(fc):
        if abs(fc) < abs(best[1]):
            best[0], best[1] = c, abs(fc)
    if math.isfinite(fd):
        if abs(fd) < abs(best[1]):
            best[0], best[1] = d, abs(fd)

    for _ in range(max_iter):
        if call.spent() or call.expired():
            break
        if abs(fc) < abs(fd):
            b = d
            d = c
            fd = fc
            c = b - phi * (b - a)
            fc, _ = _safe_eval(call, c)
            if math.isfinite(fc):
                if abs(fc) < abs(best[1]):
                    best[0], best[1] = c, abs(fc)
        else:
            a = c
            c = d
            fc = fd
            d = a + phi * (b - a)
            fd, _ = _safe_eval(call, d)
            if math.isfinite(fd):
                if abs(fd) < abs(best[1]):
                    best[0], best[1] = d, abs(fd)

    return best[0], best[1]


def _find_bracket(call, x0, fx0, best, max_expand=80):
    """
    Search for a bracket [a, b] with f(a)*f(b) < 0.
    Uses geometric expansion around x0.
    Returns (a, b, fa, fb) or None.
    """
    if not math.isfinite(fx0):
        return None

    # First, check if x0 is already a root
    if abs(fx0) <= TOL_TIGHT:
        return (x0, x0, fx0, fx0)

    scale = max(1.0, abs(x0))
    # Start with a small step, expand geometrically
    # Use a mix of linear and exponential expansion
    steps = []
    for k in range(1, max_expand + 1):
        # Alternate between small and large steps to catch nearby and distant roots
        if k <= 10:
            step = k * 1e-3 * scale
        elif k <= 30:
            step = (k - 10) * 1e-2 * scale
        elif k <= 60:
            step = (k - 30) * 1e-1 * scale
        else:
            step = (k - 60) * 1.0 * scale + 3.0 * scale

        steps.append(step)

    # Evaluate points on both sides
    points = [(x0, fx0)]  # (x, fx)
    for step in steps:
        if call.spent() or call.expired():
            break
        x_plus = x0 + step
        x_minus = x0 - step

        fp, okp = _safe_eval(call, x_plus)
        if okp:
            points.append((x_plus, fp))
            if abs(fp) < abs(best[1]):
                best[0], best[1] = x_plus, abs(fp)

        fm, okm = _safe_eval(call, x_minus)
        if okm:
            points.append((x_minus, fm))
            if abs(fm) < abs(best[1]):
                best[0], best[1] = x_minus, abs(fm)

    # Now check all pairs for sign change
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            xi, fi = points[i]
            xj, fj = points[j]
            if math.isfinite(fi) and math.isfinite(fj) and fi * fj < 0:
                # Found a bracket
                if fi > 0:
                    return (xj, xi, fj, fi)
                else:
                    return (xi, xj, fi, fj)

    return None


def _bisection(call, a, b, fa, fb, best, max_iter=60):
    """
    Bisection method to find root in [a, b].
    Assumes f(a)*f(b) < 0.
    Updates best with the best point found.
    Returns (x_root, f_root) or None if failed.
    """
    for _ in range(max_iter):
        if call.spent() or call.expired():
            break

        mid = (a + b) / 2.0
        fm, okm = _safe_eval(call, mid)
        if not okm:
            break

        if abs(fm) < abs(best[1]):
            best[0], best[1] = mid, abs(fm)

        if abs(fm) <= TOL_TIGHT:
            return (mid, fm)

        if fa * fm <= 0:
            b = mid
            fb = fm
        else:
            a = mid
            fa = fm

        # Check if interval is small enough
        if abs(b - a) < 1e-12 * max(1.0, abs(mid)):
            return (mid, fm)

    return (best[0], best[1])


def find_root(f, x0):
    call = _EvalBudget(f, cap=EVAL_CAP, time_limit=TIME_LIMIT)

    # Best point seen so far
    best = [float(x0), math.inf]

    # Evaluate the initial guess
    fx0, ok0 = _safe_eval(call, float(x0))
    if ok0:
        if abs(fx0) < abs(best[1]):
            best[0], best[1] = float(x0), abs(fx0)
        if abs(fx0) <= TOL_TIGHT:
            return float(x0)

    # Phase 1: Geometric expansion to find a bracket
    bracket = _find_bracket(call, float(x0), fx0, best)

    if bracket is not None:
        a, b, fa, fb = bracket
        if a != b:  # Not a trivial root
            result = _bisection(call, a, b, fa, fb, best)
            if result is not None:
                x_root, f_root = result
                if math.isfinite(f_root) and abs(f_root) <= TOL:
                    return float(x_root)
                # Update best if we found something better
                if math.isfinite(f_root) and abs(f_root) < abs(best[1]):
                    best[0], best[1] = x_root, abs(f_root)

    # Phase 2: If no bracket found, try golden-section minimization over
    # an expanding window to find a local minimum of |f|, then search for a
    # bracket around that minimum.
    if not call.spent() and not call.expired():
        x0_val = float(x0)
        scale = max(1.0, abs(x0_val))

        # Try several windows of increasing size
        for window_factor in [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
            if call.spent() or call.expired():
                break

            window = window_factor * scale
            a_win = x0_val - window
            b_win = x0_val + window

            # Minimize |f| on this window
            _golden_section_minimize(call, a_win, b_win, best, max_iter=20)

            # If we found a very small |f|, try to bracket around it
            if best[1] < 1e-4:
                # Local search around the best point
                x_best = best[0]
                local_scale = max(1e-6, window * 1e-3)

                # Try small steps around x_best
                for k in range(1, 20):
                    if call.spent() or call.expired():
                        break
                    step = k * local_scale

                    x1 = x_best - step
                    x2 = x_best + step

                    f1, ok1 = _safe_eval(call, x1)
                    f2, ok2 = _safe_eval(call, x2)

                    if ok1 and ok2 and f1 * f2 < 0:
                        # Found a bracket!
                        if f1 > 0:
                            a_b, b_b, fa_b, fb_b = x2, x1, f2, f1
                        else:
                            a_b, b_b, fa_b, fb_b = x1, x2, f1, f2

                        result = _bisection(call, a_b, b_b, fa_b, fb_b, best, max_iter=60)
                        if result is not None:
                            x_root, f_root = result
                            if math.isfinite(f_root) and abs(f_root) <= TOL:
                                return float(x_root)
                            if math.isfinite(f_root) and abs(f_root) < abs(best[1]):
                                best[0], best[1] = x_root, abs(f_root)
                        break  # Stop local search after first bracket attempt

                    # Also check if x_best itself is a root
                    if ok1 and abs(f1) < abs(best[1]):
                        best[0], best[1] = x1, abs(f1)
                    if ok2 and abs(f2) < abs(best[1]):
                        best[0], best[1] = x2, abs(f2)

                if best[1] <= TOL:
                    return float(best[0])

    # Return the best point found
    return float(best[0])
