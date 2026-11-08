import time
import numpy as np

_START = time.time()
_DEADLINE = 580.0  # safety margin under the 630s kill

def _elapsed():
    return time.time() - _START

def _laguerre_double_roots():
    # Canonical 12 positive double-root locations for the m=12, n=1 Laguerre construction:
    # the roots of the standard Laguerre polynomial L_12(x).
    from scipy.special import roots_laguerre
    x, _ = roots_laguerre(12)
    x = np.asarray(x, dtype=float)
    x = np.sort(x)
    # Root constraints: positive, strictly increasing.
    x = x[x > 0.0]
    return x.tolist()

def _valid(roots):
    if len(roots) != 12:
        return False
    for i, r in enumerate(roots):
        if not np.isfinite(r) or r <= 0.0:
            return False
        if i > 0 and r <= roots[i-1]:
            return False
    return True

def construct():
    best = _laguerre_double_roots()
    if not _valid(best):
        # Fallback: a valid, ordered, positive spread (should not trigger).
        best = [0.5 * (i + 1) for i in range(12)]
    return best
