"""Refined Cohn-Goncalves Laguerre-polynomial construction for the Fourier
sign uncertainty upper bound.

STRUCTURAL CHANGE: The previous approaches used $\alpha=0$ (ordinary) and
$\alpha=1$ (generalized) Laguerre polynomials. This program tests
$\alpha=-0.5$, which corresponds to a different weight function
$x^{-0.5}e^{-x}$, often arising in problems with specific radial or
boundary conditions (e.g., associated with half-integer degrees or
specific Fourier transforms on the half-line). The roots are computed
via the Jacobi matrix eigenvalue method and refined with Newton's method.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 630.0
_SAFETY = 30.0
_M = 12
_N = 1
_ALPHA = -0.5  # Structural change: try alpha = -0.5


def _deadline_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY)


def _laguerre_zeros(deg, alpha):
    """Return the positive zeros of the generalized Laguerre polynomial L_deg^(alpha)."""
    n = deg
    # Jacobi matrix for generalized Laguerre L_n^(alpha)
    # diag: 2k + alpha + 1
    # off: sqrt(k * (k + alpha))
    k = np.arange(1, n + 1, dtype=float)
    diag = 2.0 * k + alpha + 1.0
    off = np.sqrt(np.arange(1, n, dtype=float) * (np.arange(1, n, dtype=float) + alpha))
    
    # Check for valid sqrt (k + alpha must be positive)
    if np.any(np.arange(1, n, dtype=float) + alpha <= 0):
        # Fallback to alpha=0 if alpha is too negative for the off-diagonal
        return _laguerre_zeros(deg, 0.0)
        
    jac = np.diag(diag, 0) + np.diag(off, 1) + np.diag(off, -1)
    zeros = np.linalg.eigvalsh(jac)
    zeros = np.sort(zeros)
    return zeros


def construct():
    """Return a list of 12 positive double-root locations for the Laguerre
    combination (m=12, n=1)."""
    best = None
    try:
        roots = _laguerre_zeros(_M, _ALPHA)
        
        if not np.all(roots > 0.0):
            raise ValueError("non-positive root")
        if not np.all(np.diff(roots) > 0.0):
            raise ValueError("not strictly ordered")
        if len(roots) != _M:
            raise ValueError("wrong count")
            
        best = [float(r) for r in roots]
        
        if not _deadline_exceeded():
            refined = _newton_refine(roots, alpha=_ALPHA)
            if refined is not None:
                best = refined
                
        return best
        
    except Exception:
        if best is not None:
            return best
        # Fallback to alpha=0 (known good score 0.5212)
        try:
            roots = _laguerre_zeros(_M, 0.0)
            if np.all(roots > 0.0) and np.all(np.diff(roots) > 0.0):
                return [float(r) for r in roots]
        except Exception:
            pass
        return [4.0 * (i + 1) for i in range(_M)]


def _laguerre_val(x, k, alpha):
    """Evaluate L_k^(alpha)(x) via recurrence."""
    if k == 0:
        return 1.0
    if k == 1:
        return 1.0 + alpha - x
    l0 = 1.0
    l1 = 1.0 + alpha - x
    for n in range(1, k):
        l2 = ((2 * n + alpha + 1 - x) * l1 - (n + alpha) * l0) / (n + 1)
        l0, l1 = l1, l2
    return l1


def _laguerre_deriv(x, k, alpha):
    """Derivative of L_k^(alpha)(x) = -L_{k-1}^{(alpha+1)}(x)."""
    if k == 0:
        return 0.0
    a = alpha + 1.0
    n = k - 1
    if n == 0:
        return -1.0
    lm1 = 1.0
    lm = 1.0 + a - x
    for m in range(1, n):
        lp1 = ((2 * m + a + 1 - x) * lm - (m + a) * lm1) / (m + 1)
        lm1, lm = lm, lp1
    return -lm


def _newton_refine(roots, max_iter=50, tol=1e-13, alpha=0.0):
    """Refine roots using Newton's method."""
    out = []
    for r in roots:
        x = float(r)
        for _ in range(max_iter):
            f = _laguerre_val(x, _M, alpha)
            df = _laguerre_deriv(x, _M, alpha)
            if abs(df) < 1e-300:
                break
            step = f / df
            x -= step
            if abs(step) < tol:
                break
        if not (x > 0.0):
            return None
        out.append(x)
    out = sorted(out)
    if not np.all(np.diff(out) > 0.0):
        return None
    return [float(v) for v in out]
# EVOLVE-BLOCK-END
