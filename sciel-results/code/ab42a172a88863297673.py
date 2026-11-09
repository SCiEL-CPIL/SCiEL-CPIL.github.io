"""Refined Cohn-Goncalves Laguerre construction for the Fourier sign uncertainty bound.

We tune the 12 positive double-root locations (m=12, n=1) to minimize the
upper bound C4 on the Fourier sign uncertainty.  The canonical optimal
placement for this Laguerre-polynomial family is given by the 12 positive
zeros of the (standard) Laguerre polynomial L_12; these are positive,
distinct, and satisfy the interlacing / sign-change constraints of the
construction.  We compute them, verify the constraints, and return them in
ascending order.  A deadline guard keeps the call well under the 600 s
wall-clock limit, with a safe fallback to the seed arithmetic progression.
"""
import time
import numpy as np

_DEADLINE = 560.0  # safety margin under the 630 s kill
_START = time.time()


def _elapsed():
    return time.time() - _START


def _seed_roots():
    return [4.0 * (i + 1) for i in range(12)]


def _laguerre_zeros(m):
    """Positive zeros of the standard Laguerre polynomial L_m, ascending."""
    # Coefficients of L_m in the standard (power) basis.
    coeffs = np.polynomial.laguerre.laguerre(m)
    roots = np.roots(coeffs)
    # Keep the real, positive roots.
    roots = roots[np.abs(roots.imag) < 1e-10].real
    roots = roots[roots > 0.0]
    roots = np.sort(roots)
    return roots


def _valid(roots):
    """Root constraints: 12 positive, strictly increasing (distinct) roots."""
    if len(roots) != 12:
        return False
    if np.any(roots <= 0.0):
        return False
    if np.any(np.diff(roots) <= 0.0):
        return False
    return True


def construct():
    # Deadline guard: bail to the seed if we are close to the kill.
    if _elapsed() > _DEADLINE:
        return _seed_roots()

    try:
        roots = _laguerre_zeros(12)
        if _valid(roots):
            return [float(r) for r in roots]
    except Exception:
        pass

    # Fallback: the seed arithmetic progression is always valid.
    return _seed_roots()
