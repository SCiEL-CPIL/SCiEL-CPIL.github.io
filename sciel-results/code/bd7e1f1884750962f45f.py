"""Search for coefficients minimizing the outermost sign-changing root of q=g/x^2."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 180.0  # safety margin under 210s kill


def _coeffs_to_P(c):
    """Return coefficient array for P(y) = q(sqrt(y)) as a degree-5 poly in y.
    P(y) = c3 y^5 + (c2+660c3) y^4 + (c1-24c2-60c3) y^3
           + (-6c1+72c2+3960c3) y^2 + (-6c1-32c2-1512c3) y
    """
    c0, c1, c2 = c
    c3 = -(c0 + 3.0 * c1 + 15.0 * c2) / 105.0
    a5 = c3
    a4 = c2 + 660.0 * c3
    a3 = c1 - 24.0 * c2 - 60.0 * c3
    a2 = -6.0 * c1 + 72.0 * c2 + 3960.0 * c3
    a1 = -6.0 * c1 - 32.0 * c2 - 1512.0 * c3
    return np.array([a5, a4, a3, a2, a1, 0.0])


def _objective(c):
    """Largest positive real root of P(y); return it (minimize this).
    If no positive root or numerically bad, return a large penalty."""
    try:
        P = _coeffs_to_P(c)
        # drop trailing zero
        roots = np.roots(P)
        pos = roots[np.real(roots) > 1e-9]
        pos = np.real(pos)
        if pos.size == 0:
            return 1e6
        return float(np.max(pos))
    except Exception:
        return 1e6


def construct():
    start = time.time()
    best_c = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _objective(best_c)

    def remaining():
        return _DEADLINE - (time.time() - start)

    # Multi-start local optimization to reduce the outermost root.
    rng = np.random.default_rng(0)
    starts = [best_c.copy()]
    # a few deterministic-ish perturbations around the seed
    for scale in (0.1, 0.5, 1.0, 2.0):
        for _ in range(4):
            p = best_c + scale * rng.standard_normal(3)
            starts.append(p)

    for s in starts:
        if remaining() < 5.0:
            break
        try:
            res = minimize(
                _objective,
                s,
                method="Nelder-Mead",
                options={"maxiter": 400, "xatol": 1e-9, "fatol": 1e-12,
                         "adaptive": True},
            )
            if remaining() < 3.0:
                break
            if res.fun < best_val:
                best_val = float(res.fun)
                best_c = np.array(res.x, dtype=float)
        except Exception:
            continue

    return np.array(best_c, dtype=float)
# EVOLVE-BLOCK-END
