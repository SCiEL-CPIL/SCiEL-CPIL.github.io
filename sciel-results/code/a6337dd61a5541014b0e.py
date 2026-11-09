"""Root finding for ill-conditioned equations: robust adaptive hybrid."""
import time
import math
import numpy as np
from scipy.optimize import brentq, newton

# Global deadline guard (wall-clock)
_DEADLINE = time.time() + 140.0  # 140s to be safe within 150s limit
_MAX_EVALS = 500

class _EvalBudget:
    def __init__(self, max_evals=_MAX_EVALS):
        self.max_evals = max_evals
        self.count = 0
        self.expired = False

    def spent(self):
        return self.count

    def check(self):
        if self.count >= self.max_evals:
            self.expired = True
        if time.time() > _DEADLINE:
            self.expired = True
        return not self.expired

    def consume(self, n=1):
        self.count += n
        return self.check()

    def expired_flag(self):
        return self.expired


def _wrap(f, budget):
    """Wrap f to count evaluations and check deadline."""
    def wrapped(x):
        if not budget.check():
            return 0.0
        val = f(x)
        budget.consume(1)
        return val
    return wrapped


def _safe_newton(f, x0, budget, max_iter=50):
    """Safeguarded Newton with finite-difference derivative."""
    x = float(x0)
    best_x = x
    best_f = abs(f(x))
    budget.consume(1)  # for initial f(x)

    for i in range(max_iter):
        if not budget.check():
            break

        fx = f(x)
        budget.consume(1)
        if abs(fx) < best_f:
            best_f = abs(fx)
            best_x = x
        if abs(fx) <= 1e-12:
            return x, best_x, best_f

        # Central difference if possible
        h = 1e-5 * max(1.0, abs(x))
        if budget.check():
            fp = f(x + h)
            budget.consume(1)
            if budget.check():
                fm = f(x - h)
                budget.consume(1)
                d = (fp - fm) / (2 * h)
            else:
                d = (fp - fx) / h
        else:
            break

        if abs(d) < 1e-15:
            break

        step = fx / d
        # Safeguard: limit step size
        max_step = max(1.0, abs(x))
        if abs(step) > max_step:
            step = math.copysign(max_step, step)

        x_new = x - step
        if not math.isfinite(x_new):
            break

        x = x_new

    return x, best_x, best_f


def _try_brent(f, a, b, budget, max_iter=100):
    """Try Brent's method on [a, b]. Returns (success, x, best_x, best_f)."""
    fa = f(a)
    budget.consume(1)
    fb = f(b)
    budget.consume(1)

    if not budget.check():
        return False, a, a, abs(fa)

    if fa == 0:
        return True, a, a, 0.0
    if fb == 0:
        return True, b, b, 0.0

    if fa * fb > 0:
        return False, a, a, min(abs(fa), abs(fb))

    try:
        x = brentq(f, a, b, maxiter=max_iter, xtol=1e-14, rtol=1e-14)
        budget.consume(1)  # approximate; brentq uses multiple evals internally
        # Actually brentq calls f multiple times, so our budget tracking is approximate
        fx = f(x)
        budget.consume(1)
        return True, x, x, abs(fx)
    except Exception:
        return False, a, a, min(abs(fa), abs(fb))


def _bracket(f, x0, budget):
    """Try to find a bracket [a, b] with sign change around x0."""
    a = x0
    fa = f(a)
    budget.consume(1)
    if abs(fa) <= 1e-12:
        return a, a, fa

    # Expand bracket
    step = 1e-2 * max(1.0, abs(x0))
    for i in range(30):
        if not budget.check():
            break
        b = x0 + step
        fb = f(b)
        budget.consume(1)
        if fa * fb <= 0:
            return a, b, fa
        a = b
        fa = fb
        step *= 2.0

        b2 = x0 - step
        fb2 = f(b2)
        budget.consume(1)
        if fa * fb2 <= 0:
            return b2, a, fb2
        # Keep the side with smaller |f| as new anchor
        if abs(fb2) < abs(fa):
            a = b2
            fa = fb2

    return None, None, None


def find_root(f, x0):
    """
    Find x with |f(x)| <= 1e-8 using at most 500 evaluations of f.
    Returns x (float).
    """
    budget = _EvalBudget(_MAX_EVALS)
    fw = _wrap(f, budget)

    # Initial evaluation
    x0 = float(x0)
    try:
        fx0 = fw(x0)
    except Exception:
        return x0

    if abs(fx0) <= 1e-8:
        return x0

    best_x = x0
    best_f = abs(fx0)

    # Phase 1: Safeguarded Newton
    x, best_x_n, best_f_n = _safe_newton(fw, x0, budget, max_iter=40)
    if best_f_n < best_f:
        best_x = best_x_n
        best_f = best_f_n
    if best_f <= 1e-8:
        return best_x

    # Phase 2: Try to bracket and use Brent
    a, b, fa = _bracket(fw, best_x, budget)
    if a is not None and b is not None:
        success, x_brent, best_x_b, best_f_b = _try_brent(fw, a, b, budget)
        if best_f_b < best_f:
            best_x = best_x_b
            best_f = best_f_b
        if success and best_f <= 1e-8:
            return best_x

    # Phase 3: Try Newton from a few perturbed starting points
    if budget.check():
        perturbations = [
            x0 * (1 + 1e-3) if x0 != 0 else 1e-3,
            x0 * (1 - 1e-3) if x0 != 0 else -1e-3,
            x0 + 1e-2,
            x0 - 1e-2,
            x0 * 1.01 if x0 != 0 else 0.01,
            x0 * 0.99 if x0 != 0 else -0.01,
        ]
        for px in perturbations:
            if not budget.check():
                break
            try:
                x_p, best_x_p, best_f_p = _safe_newton(fw, px, budget, max_iter=20)
                if best_f_p < best_f:
                    best_x = best_x_p
                    best_f = best_f_p
                if best_f <= 1e-8:
                    return best_x
            except Exception:
                pass

    # Phase 4: If we have a bracket, try again with more effort
    if not budget.expired_flag() and a is not None and b is not None:
        try:
            x_brent2 = brentq(fw, a, b, maxiter=50, xtol=1e-15, rtol=1e-15)
            fx_b = fw(x_brent2)
            if abs(fx_b) < best_f:
                best_x = x_brent2
                best_f = abs(fx_b)
            if best_f <= 1e-8:
                return best_x
        except Exception:
            pass

    return best_x
