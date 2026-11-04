"""Second autocorrelation inequality – maximise the C2 lower-bound ratio.

Design: two disjoint Gaussian lobes whose offset and width are tuned to keep
the autoconvolution peak low while preserving L2 norm.  We use a small
multi-start local optimisation with a hard wall-clock deadline guard.
"""
import time
import numpy as np

N = 50
_DEADLINE = 115.0  # seconds; safety margin before the 120 s kill switch
_START = time.monotonic()


def _remaining():
    return _DEADLINE - (time.monotonic() - _START)


def _score(h):
    """Return the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf).

    h is a non-negative array of length N.  The autoconvolution is the
    discrete convolution h * h (full length 2N-1); we take its supremum.
    If the denominator is zero or negative we return 0.
    """
    h = np.asarray(h, dtype=float)
    l2_sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    sup = float(np.max(conv))
    if sup <= 0.0:
        return 0.0
    return l2_sq / (l1 * sup)


def _two_lobe(delta, sigma):
    """Build a 50-sample profile: two Gaussian lobes at +/-delta from centre."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    # Centre of interval is 0.  Lobes at +/-delta.
    h = np.exp(-((x - delta) ** 2) / (2.0 * sigma ** 2)) + \
        np.exp(-((x + delta) ** 2) / (2.0 * sigma ** 2))
    # Clip to [0, 1] for numerical safety and to keep values nonnegative.
    h = np.clip(h, 0.0, None)
    return h


def _neg_score(params):
    delta, sigma = params
    delta = max(abs(delta), 1e-6)
    sigma = max(sigma, 1e-4)
    h = _two_lobe(delta, sigma)
    return -_score(h)


def construct():
    """Return 50 nonnegative step heights maximising the C2 ratio."""
    from scipy.optimize import minimize

    best_h = _two_lobe(0.08, 0.04)
    best_val = _score(best_h)

    # Multi-start grid over (delta, sigma)
    deltas = np.linspace(0.02, 0.22, 9)
    sigmas = np.linspace(0.01, 0.10, 9)

    # First pass: evaluate the coarse grid to find promising starts
    grid_best_val = best_val
    grid_best_params = None
    for d in deltas:
        if _remaining() < 10.0:
            break
        for s in sigmas:
            if _remaining() < 5.0:
                break
            p = (d, s)
            v = -_neg_score(p)
            if v > grid_best_val:
                grid_best_val = v
                grid_best_params = p
                h = _two_lobe(d, s)
                if v > best_val:
                    best_val = v
                    best_h = h

    # Second pass: local refinement around the best grid points
    if grid_best_params is not None:
        starts = [grid_best_params]
        # Add a few neighbouring starts for diversity
        d0, s0 = grid_best_params
        for dd in (-0.02, 0.02):
            for ss in (-0.015, 0.015):
                nd = max(d0 + dd, 0.005)
                ns = max(s0 + ss, 0.005)
                starts.append((nd, ns))
    else:
        starts = [(d, s) for d in deltas[::3] for s in sigmas[::3]][:6]

    for p0 in starts:
        if _remaining() < 15.0:
            break
        try:
            res = minimize(
                _neg_score,
                x0=np.array(p0, dtype=float),
                method='L-BFGS-B',
                bounds=[(0.005, 0.24), (0.005, 0.15)],
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8},
            )
            if _remaining() < 5.0:
                break
            val = -res.fun
            h = _two_lobe(res.x[0], res.x[1])
            if val > best_val:
                best_val = val
                best_h = h
        except Exception:
            continue

    # Convert to plain Python floats (nonnegative)
    out = [float(max(0.0, v)) for v in best_h]
    return out
