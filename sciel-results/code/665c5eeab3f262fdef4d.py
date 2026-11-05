"""Portfolio root finder: robust, ill-conditioned-aware, with deadline guard."""
import time
import math

_TOL = 1e-8
_MAX_EVALS = 500
# Safety margin well under the 150s kill; leave headroom for overhead.
_DEADLINE = time.monotonic() + 140.0


def _alive():
    return time.monotonic() < _DEADLINE


def _is_finite(v):
    try:
        return math.isfinite(v)
    except (TypeError, ValueError):
        return False


class _EvalCounter:
    def __init__(self, f, max_evals=500):
        self._f = f
        self._count = 0
        self._max = max_evals
        self._cache = {}

    def __call__(self, x):
        x = float(x)
        if self._count >= self._max:
            return 0.0
        # Use a small cache to avoid re-evaluating same point within a method
        key = x
        if key in self._cache:
            return self._cache[key]
        try:
            val = self._f(x)
            val = float(val)
        except Exception:
            val = 0.0
        if not _is_finite(val):
            val = 0.0
        self._count += 1
        self._cache[key] = val
        return val

    @property
    def count(self):
        return self._count


def _finite_diff(fc, x, fx, h_base=1e-5):
    """Central-difference derivative; returns (deriv, evals_used)."""
    base = 1.0 + abs(x)
    for h in (h_base, h_base * 10.0, h_base * 0.1):
        hh = h * base
        xp = x + hh
        xm = x - hh
        fp = fc(xp)
        fm = fc(xm)
        d = (fp - fm) / (2.0 * hh)
        if d != 0.0 and abs(d) > 1e-300 and _is_finite(d):
            return d, 2
    # forward difference fallback
    hh = h_base * base
    fp = fc(x + hh)
    d = (fp - fx) / hh
    return d, 1


def _newton(fc, x, max_iter=60):
    """Newton with finite-diff derivative and damping."""
    x = float(x)
    for _ in range(max_iter):
        if not _alive():
            return x
        fx = fc(x)
        if abs(fx) <= _TOL:
            return x
        d, _ = _finite_diff(fc, x, fx)
        if d == 0.0 or not _is_finite(d):
            break
        step = fx / d
        # Damp if step too large
        if abs(step) > 10.0 * max(1.0, abs(x)):
            step = math.copysign(10.0 * max(1.0, abs(x)), step)
        x_new = x - step
        if not _is_finite(x_new):
            break
        x = x_new
    return x


def _secant(fc, x0, x1, max_iter=60):
    """Secant method."""
    x = float(x0)
    y = float(x1)
    fx = fc(x)
    fy = fc(y)
    for _ in range(max_iter):
        if not _alive():
            return x if abs(fx) <= abs(fy) else y
        if abs(fx) <= _TOL:
            return x
        if abs(fy) <= _TOL:
            return y
        if fy == fx:
            break
        x_new = x - fx * (x - y) / (fx - fy)
        if not _is_finite(x_new):
            break
        x, y = y, x_new
        fx, fy = fy, fc(y)
    return x if abs(fx) <= abs(fy) else y


def _bisect(fc, a, b, max_iter=100):
    """Bisection with Illinois acceleration."""
    fa = fc(a)
    fb = fc(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0:
        return a if abs(fa) < abs(fb) else b
    for _ in range(max_iter):
        if not _alive():
            mid = (a + b) / 2.0
            return mid
        mid = (a + b) / 2.0
        fm = fc(mid)
        if abs(fm) <= _TOL:
            return mid
        if fa * fm <= 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
            fa *= 0.5  # Illinois
    return (a + b) / 2.0


def _find_bracket(fc, x0, max_evals=20):
    """Try to find a bracket around x0."""
    x = float(x0)
    fx = fc(x)
    if abs(fx) <= _TOL:
        return (x, x)
    step = 1e-3
    for _ in range(max_evals // 2):
        if not _alive() or fc.count >= _MAX_EVALS:
            break
        xp = x + step
        fp = fc(xp)
        if fx * fp < 0:
            return (x, xp)
        xm = x - step
        fm = fc(xm)
        if fx * fm < 0:
            return (xm, x)
        step *= 2.0
        if step > 1e10:
            break
    return None


def _muller(fc, x0, x1, x2, max_iter=40):
    """Muller's method for three points."""
    a, b, c = float(x0), float(x1), float(x2)
    fa, fb, fc_ = fc(a), fc(b), fc(c)
    for _ in range(max_iter):
        if not _alive():
            # return best of three
            best = a
            if abs(fb) < abs(fa):
                best = b
            if abs(fc_) < abs(fc(best)):
                best = c
            return best
        if abs(fa) <= _TOL:
            return a
        if abs(fb) <= _TOL:
            return b
        if abs(fc_) <= _TOL:
            return c
        # Muller's formula
        d1 = (fb - fa) / (b - a)
        d2 = (fc_ - fb) / (c - b)
        denom = d2 - d1
        if abs(denom) < 1e-300:
            # fall back to secant
            break
        # p = -2*fb*denom / ( (fb-fa)/((b-a)*(d2-d1)) ... ) -- use standard form
        # Numerator: -2 * fb * (d2 - d1)
        # Denominator: (fb - fa) / (b - a) * (d2 - d1) ... actually:
        # Let's use the standard Muller:
        # x_{n+1} = x_n - 2*f_n*(x_n - x_{n-1}) / ( (f_n - f_{n-1})/(x_n - x_{n-1}) - (f_n - f_{n-2})/(x_n - x_{n-2}) ) * (x_n - x_{n-1}) ...
        # Simpler: use the quadratic fit approach
        # We want the root of the quadratic through (a,fa), (b,fb), (c,fc_)
        # Use Lagrange or Newton on the quadratic
        # Coefficients of quadratic: A*x^2 + B*x + C
        # Using divided differences:
        # f[a,b] = (fb-fa)/(b-a) = d1
        # f[b,c] = (fc_-fb)/(c-b) = d2
        # f[a,b,c] = (d2-d1)/(c-a)
        fabc = (d2 - d1) / (c - a)
        # Quadratic: q(t) = fb + (t-b)*d1 + (t-a)(t-b)*fabc
        # q(t) = fabc*t^2 + (d1 - fabc*(a+b))*t + (fb - d1*b + fabc*a*b)
        A = fabc
        B = d1 - fabc * (a + b)
        C = fb - d1 * b + fabc * a * b
        # Solve A*t^2 + B*t + C = 0, pick root closer to b
        disc = B * B - 4.0 * A * C
        if disc < 0:
            disc = 0.0
        sq = math.sqrt(disc)
        # Two roots: (-B + sq)/(2A) and (-B - sq)/(2A)
        # Pick the one with larger denominator magnitude to avoid cancellation
        if B >= 0:
            t = (-B - sq) / (2.0 * A)
        else:
            t = (-B + sq) / (2.0 * A)
        if not _is_finite(t):
            break
        # Shift to avoid overflow
        t = b + (t - b)
        if not _is_finite(t):
            break
        ft = fc(t)
        a, b, c = b, c, t
        fa, fb, fc_ = fb, fc_, ft
    # Return best
    best = a
    if abs(fb) < abs(fa):
        best = b
    if abs(fc_) < abs(fc(best)):
        best = c
    return best


def _polish(fc, x, max_iter=30):
    """High-precision polish using Newton with very small h."""
    x = float(x)
    for _ in range(max_iter):
        if not _alive():
            return x
        fx = fc(x)
        if abs(fx) <= 1e-12:
            return x
        d, _ = _finite_diff(fc, x, fx, h_base=1e-7)
        if d == 0.0 or not _is_finite(d):
            break
        step = fx / d
        if abs(step) < 1e-15:
            break
        x_new = x - step
        if not _is_finite(x_new):
            break
        x = x_new
    return x


def find_root(f, x0):
    fc = _EvalCounter(f, _MAX_EVALS)
    x0 = float(x0)

    # First check if x0 is already a root
    fx0 = fc(x0)
    if abs(fx0) <= _TOL:
        return x0

    # Try Newton first
    x_newton = _newton(fc, x0, max_iter=50)
    if abs(fc(x_newton)) <= _TOL:
        return _polish(fc, x_newton)

    # Try to find a bracket and use bisection
    bracket = _find_bracket(fc, x0, max_evals=20)
    if bracket is not None:
        a, b = bracket
        x_bisect = _bisect(fc, a, b, max_iter=80)
        if abs(fc(x_bisect)) <= _TOL:
            return _polish(fc, x_bisect)

    # Try secant with a perturbed point
    h = 1e-4 * max(1.0, abs(x0))
    x_secant = _secant(fc, x0, x0 + h, max_iter=50)
    if abs(fc(x_secant)) <= _TOL:
        return _polish(fc, x_secant)

    # Try Muller with three points
    h1 = 1e-3 * max(1.0, abs(x0))
    h2 = 2e-3 * max(1.0, abs(x0))
    x_muller = _muller(fc, x0 - h1, x0, x0 + h2, max_iter=30)
    if abs(fc(x_muller)) <= _TOL:
        return _polish(fc, x_muller)

    # Return the best result found so far
    candidates = [x0, x_newton, x_secant, x_muller]
    if bracket is not None:
        candidates.append((bracket[0] + bracket[1]) / 2.0)
    best = x0
    best_val = abs(fx0)
    for c in candidates:
        try:
            val = abs(fc(c))
            if val < best_val:
                best_val = val
                best = c
        except Exception:
            pass
    return best
