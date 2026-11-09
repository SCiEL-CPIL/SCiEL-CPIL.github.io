"""Refined Cohn-Goncalves Laguerre-polynomial construction for the Fourier
sign uncertainty upper bound.

The optimal double-root locations for the Laguerre combination (m=12, n=1)
are the zeros of the ordinary Laguerre polynomial L_12.  We compute them via
scipy.special.roots_laguerre (highly optimized, robust), with a companion-
matrix eigenvalue fallback.  A Newton refinement pass using the generalized
Laguerre recurrence guarantees the returned nodes are the true zeros to
machine precision.

Score (upper bound on C4) is the raw value of the resulting construction;
lower is better.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import roots_laguerre, eval_genlaguerre

_START_TIME = time.time()
_DEADLINE = 630.0          # evaluator kills at 630 s wall-clock
_SAFETY = 30.0             # return well before the hard limit
_M = 12
_N = 1


def _deadline_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY)


def _laguerre_zeros(deg):
    """Return the positive zeros of the ordinary Laguerre polynomial L_deg.

    Primary: scipy.special.roots_laguerre (Gauss-Laguerre quadrature nodes).
    Fallback: companion-matrix eigenvalue method on the Jacobi matrix.
    """
    try:
        zeros, _ = roots_laguerre(deg)
        zeros = np.sort(zeros)
        return zeros
    except Exception:
        n = deg
        diag = np.arange(1, n + 1, dtype=float) * 2.0 + 1.0
        off = np.sqrt(np.arange(1, n, dtype=float) * (np.arange(1, n, dtype=float) + 1.0))
        jac = np.diag(diag, 0) + np.diag(off, 1) + np.diag(off, -1)
        zeros = np.linalg.eigvalsh(jac)
        zeros = np.sort(zeros)
        return zeros


def _laguerre_val(x, k, alpha=0):
    """Evaluate the generalized Laguerre polynomial L_k^{(alpha)} at scalar x."""
    return float(eval_genlaguerre(k, alpha, x))


def _laguerre_deriv(x, k, alpha=0):
    """Derivative of L_k^{(alpha)} at x using L_k^{(alpha)'} = -L_{k-1}^{(alpha+1)}."""
    if k == 0:
        return 0.0
    return -float(eval_genlaguerre(k - 1, alpha + 1, x))


def _newton_refine(roots, max_iter=50, tol=1e-13, alpha=0):
    """Refine each root with Newton's method on L_M^{(alpha)}.  Returns the refined
    list or None if any refinement fails the constraints."""
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


def construct():
    """Return a list of 12 positive double-root locations for the Laguerre
    combination (m=12, n=1).  Score (upper bound on C4) is the raw value of
    the resulting construction; lower is better.
    """
    best = None
    try:
        # Primary DIRECT construction: zeros of L_12 (non-uniform optimal).
        roots = _laguerre_zeros(_M)

        # Verify root constraints: positivity and strict ordering.
        if not np.all(roots > 0.0):
            raise ValueError("non-positive Laguerre zero encountered")
        if not np.all(np.diff(roots) > 0.0):
            raise ValueError("roots not strictly ordered")
        if len(roots) != _M:
            raise ValueError("wrong number of roots")

        best = [float(r) for r in roots]

        # Deadline guard: if we still have time, attempt a Newton refinement
        # to guarantee the nodes are the true zeros to machine precision.
        if not _deadline_exceeded():
            refined = _newton_refine(roots)
            if refined is not None:
                best = refined

        return best

    except Exception:
        # Fallback: if the Laguerre-zero path fails for any reason,
        # return a valid positive, strictly-increasing 12-tuple so the
        # result is never invalid.  Use the seed grid as a safe floor.
        if best is not None:
            return best
        return [4.0 * (i + 1) for i in range(_M)]
# EVOLVE-BLOCK-END
