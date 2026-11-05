"""Rewritten program: degree-5 Chebyshev-basis polynomial fit for L96 subgrid closure."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 150.0          # evaluator kills at 150 s wall-clock
_SAFETY = 5.0              # return at least this many seconds before the limit


def _cheb_to_monomial(c, degree):
    """Convert Chebyshev (T_k) coefficients to monomial coefficients.

    c[k] multiplies T_k(x).  Uses the identity
        T_k(x) = sum_{m even, m<=k} a_{k,m} x^m
    with a recurrence that is numerically stable for small degree.
    Returns monomial coefficients in ASCENDING powers of x, length degree+1.
    """
    # monomial coefficients of T_k as a list indexed by power
    # T_0 = 1, T_1 = x, T_{k} = 2 x T_{k-1} - T_{k-2}
    T = [np.zeros(degree + 1) for _ in range(degree + 1)]
    T[0][0] = 1.0
    if degree >= 1:
        T[1][1] = 1.0
    for k in range(2, degree + 1):
        T[k] = 2.0 * (T[k - 1] * np.arange(degree + 1)[None, :]) - T[k - 2]
        # multiply T[k-1] by x: shift powers
        T[k] = 2.0 * np.roll(T[k - 1], -1) - T[k - 2]
    mon = np.zeros(degree + 1)
    for k in range(degree + 1):
        mon += c[k] * T[k]
    return mon


def _fit_chebyshev(x, y, degree):
    """Least-squares fit in Chebyshev basis on x's data range.

    Returns Chebyshev coefficients (ascending T_k), length degree+1.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        # degenerate: return a constant fit
        c = np.zeros(degree + 1)
        c[0] = float(np.mean(y)) if y.size else 0.0
        return c

    # map x -> t in [-1, 1]
    t = (x - 0.5 * (hi + lo)) / (0.5 * (hi - lo))

    # build Chebyshev Vandermonde via stable recurrence
    V = np.empty((x.size, degree + 1))
    V[:, 0] = 1.0
    if degree >= 1:
        V[:, 1] = t
    for k in range(2, degree + 1):
        V[:, k] = 2.0 * t * V[:, k - 1] - V[:, k - 2]

    # solve least squares
    c, *_ = np.linalg.lstsq(V, y, rcond=None)
    return c


def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    degree = 5

    x = np.asarray(training_x, dtype=float).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=float).ravel()

    # Fallback: if inputs are empty/degenerate, return zeros (finite).
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return np.zeros(degree + 1, dtype=float)

    # Filter non-finite samples to keep the solve well-defined.
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.all(mask):
        x = x[mask]
        y = y[mask]
    if x.size < degree + 1:
        # not enough points for degree 5; fall back to as high a degree as possible
        degree = max(1, x.size - 1)

    try:
        c_cheb = _fit_chebyshev(x, y, degree)
        mon = _cheb_to_monomial(c_cheb, degree)
    except Exception:
        # Last-resort: plain numpy polyfit in monomial basis (less stable but safe).
        try:
            mon = np.polyfit(x, y, degree)  # descending powers
            mon = mon[::-1]                  # to ascending
        except Exception:
            mon = np.zeros(6, dtype=float)

    # Ensure exactly 6 finite coefficients in ascending powers.
    out = np.zeros(6, dtype=float)
    n = min(6, mon.size)
    out[:n] = mon[:n]
    out[~np.isfinite(out)] = 0.0

    # Deadline guard: we have already produced a valid result; nothing else to do.
    if time.time() - t_start > _DEADLINE - _SAFETY:
        pass  # return immediately

    return out
# EVOLVE-BLOCK-END
