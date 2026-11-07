"""Search for better coefficients via Nelder-Mead with a positivity penalty."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import eval_hermite
from scipy.optimize import minimize
from scipy.optimize import brentq

_START_TIME = time.time()
_DEADLINE = 180.0
_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

# Precompute Hermite polynomials (physicists') on a dense grid
_X = np.linspace(0.0, 40.0, 8001)
_H0 = eval_hermite(0, _X)
_H4 = eval_hermite(4, _X)
_H8 = eval_hermite(8, _X)
_H12 = eval_hermite(12, _X)
_X2 = _X * _X

# For the sign-change root, we need q(x) = g(x)/x^2. At x=0 this is singular,
# but g(0)=0 by construction, so q is well-defined. We'll work with g directly
# and find roots of g (excluding x=0), then take the largest positive one where
# g changes sign (i.e., g goes from + to - or - to +).

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _compute_value(c0, c1, c2):
    """Compute the task value for given coefficients. Returns None if invalid."""
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1008.0
    H12_0 = 10395.0

    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0

    # g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    g = c0 * _H0 + c1 * _H4 + c2 * _H8 + c3 * _H12

    # Check if g tends to negative at infinity: leading term is c3*H12
    # H12 ~ 12! / 6! * x^12 * (leading coeff of physicists H_n is (2x)^n / ... )
    # Actually physicists' Hermite: H_n(x) ~ (2x)^n for large x
    # So leading behavior: c3 * (2x)^12 dominates. If c3 < 0, g -> -inf.
    if c3 < 0:
        g = -g

    # Now g should be non-negative for a valid bound. Check positivity.
    if np.any(g < -1e-10):
        return None

    # Find the largest positive root of g where g changes sign.
    # q = g/x^2, roots of q are roots of g (except possibly x=0).
    # We look for the largest r > 0 where g changes sign.
    # Sign change means g goes from positive to negative or vice versa.

    # Find all sign changes in g on (0, 40]
    # g[0] should be 0 (by construction), so start from index 1
    sign_changes = []
    for i in range(1, len(_X) - 1):
        if _X[i] <= 0:
            continue
        if g[i] == 0:
            sign_changes.append(_X[i])
        elif g[i - 1] * g[i] < 0:
            # Sign change between i-1 and i
            try:
                root = brentq(lambda x: c0 * eval_hermite(0, x) + c1 * eval_hermite(4, x) +
                              c2 * eval_hermite(8, x) + c3 * eval_hermite(12, x),
                              _X[i - 1], _X[i], xtol=1e-12)
                sign_changes.append(root)
            except Exception:
                pass

    if not sign_changes:
        # If no sign change, the "largest real root at which q changes sign"
        # doesn't exist in the usual sense. This might mean g >= 0 everywhere
        # except at x=0, which would give r=0 (value 0) or be invalid.
        # Actually if g >= 0 for all x > 0 and g(0)=0, then there's no sign change.
        # The problem says "the largest real root of q at which q changes sign".
        # If no such root exists, the value might be 0 or undefined.
        # Let's treat it as r=0 for now (best case).
        return 0.0

    r = max(sign_changes)
    return r * r / (2.0 * np.pi)


def _objective(c):
    """Objective: value + penalty for negativity."""
    c0, c1, c2 = c
    val = _compute_value(c0, c1, c2)

    if val is None:
        # Compute penalty: how negative is g?
        H0_0 = 1.0
        H4_0 = 12.0
        H8_0 = 1008.0
        H12_0 = 10395.0
        c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
        g = c0 * _H0 + c1 * _H4 + c2 * _H8 + c3 * _H12
        if c3 < 0:
            g = -g
        min_g = np.min(g)
        if min_g < 0:
            penalty = 100.0 * (-min_g) ** 2
        else:
            penalty = 0.0
        return 10.0 + penalty

    return val


def construct():
    best_c = _SEED.copy()
    best_val = _compute_value(*_SEED)
    if best_val is None:
        best_val = float('inf')

    # Nelder-Mead search
    try:
        res = minimize(
            _objective,
            x0=_SEED,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if _time_left() < 10.0:
            return best_c

        c_new = res.x
        val_new = _compute_value(*c_new)
        if val_new is not None and val_new < best_val:
            best_c = c_new
            best_val = val_new
    except Exception:
        pass

    # Local refinement with tighter tolerance
    if _time_left() > 15.0:
        try:
            res2 = minimize(
                _objective,
                x0=best_c,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if _time_left() < 5.0:
                return best_c

            c_new2 = res2.x
            val_new2 = _compute_value(*c_new2)
            if val_new2 is not None and val_new2 < best_val:
                best_c = c_new2
                best_val = val_new2
        except Exception:
            pass

    # Multi-start from a few perturbations
    if _time_left() > 20.0:
        rng = np.random.default_rng(42)
        for _ in range(5):
            if _time_left() < 15.0:
                break
            try:
                c_start = best_c + rng.normal(0, 0.01, size=3)
                res3 = minimize(
                    _objective,
                    x0=c_start,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                c_new3 = res3.x
                val_new3 = _compute_value(*c_new3)
                if val_new3 is not None and val_new3 < best_val:
                    best_c = c_new3
                    best_val = val_new3
            except Exception:
                pass

    return np.array(best_c)
# EVOLVE-BLOCK-END
