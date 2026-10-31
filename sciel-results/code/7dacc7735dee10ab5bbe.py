"""Rewritten program: real-valued window shapes with zero-DC centering to minimize
the autoconvolution sidelobe energy (the C3 upper bound)."""
import time
import numpy as np

N = 400

_DEADLINE = 150.0
_START = time.time()


def _remaining():
    return _DEADLINE - 6.0 - (time.time() - _START)   # 6 s safety margin


def _sidelobe_energy(f):
    """C3 = sum over non-DC lags of |autoconvolution lag|.
    autoconv lag k = sum_i f_i * f_{i+k}  (linear, zero-padded)."""
    N = len(f)
    ac = np.correlate(f, f, mode='full')   # length 2N-1, center at N-1 is lag 0
    # ac[j] corresponds to lag (j - (N-1))
    ac = np.asarray(ac, dtype=float)
    c = N - 1
    # zero out DC (lag 0)
    ac2 = ac.copy()
    ac2[c] = 0.0
    return float(np.sum(np.abs(ac2)))


def _centered_window(shape, param, scale=1.0):
    """Build a real-valued window of length N, zero mean, scaled by `scale`."""
    n = np.arange(N)
    if shape == 'gaussian':
        # sigma in units of N
        sigma = param * N
        w = np.exp(-((n - (N - 1) / 2.0) ** 2) / (2.0 * sigma ** 2))
    elif shape == 'kaiser':
        beta = param
        w = np.kaiser(N, beta)
    elif shape == 'hann':
        # generalized raised cosine: w = 1 - a*cos(2pi n/(N-1))
        # a=1 -> Hann, a=0.5 -> Hamming, a=0 -> rectangular
        a = param
        w = 1.0 - a * np.cos(2.0 * np.pi * n / (N - 1.0))
    elif shape == 'flat_top':
        # near-constant with small edge rolloff, param controls rolloff
        w = np.ones(N)
        edge = int(param * N)
        if edge > 0:
            t = np.arange(edge)
            roll = 0.5 + 0.5 * np.cos(np.pi * t / edge)
            w[:edge] = roll
            w[-edge:] = roll[::-1]
    else:
        w = np.ones(N)
    # center (zero DC)
    w = w - np.mean(w)
    w = w * scale
    return w


def _optimize_shape(shape, param_grid, scale_grid):
    """Grid search over shape params and amplitude scales. Return best (val, f)."""
    best_val = np.inf
    best_f = None
    for p in param_grid:
        base = _centered_window(shape, p, scale=1.0)
        if not np.any(base):
            continue
        # energy is quadratic in scale, so best scale for a fixed shape is found
        # by trying a small set; but since centering already removes DC, the
        # sidelobe energy scales as scale^2. We want the raw value lower, so we
        # pick a scale that is valid. The task scores the raw value, so smaller
        # magnitude -> smaller C3. But f must be a "signed step function" with
        # 400 intervals; amplitude is free. We normalize to a sensible unit.
        for s in scale_grid:
            f = base * s
            val = _sidelobe_energy(f)
            if val < best_val:
                best_val = val
                best_f = f.copy()
    return best_val, best_f


def _local_refine(f0):
    """Nelder-Mead-free simple coordinate refinement on amplitude + small shifts.
    Uses scipy if available for a bounded L-BFGS on a few shape params."""
    try:
        from scipy.optimize import minimize
    except Exception:
        return f0

    def objective(x):
        # x = [shape_param, scale]
        p, s = x
        f = _centered_window(_REFINE_SHAPE, p, scale=s)
        return _sidelobe_energy(f)

    x0 = np.array([_REFINE_P0, _REFINE_S0])
    bounds = [(_REFINE_PMIN, _REFINE_PMAX), (_REFINE_SMIN, _REFINE_SMAX)]
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 400, 'maxfun': 8000})
        if res.success or res.fun < _REFINE_BEST:
            f = _centered_window(_REFINE_SHAPE, res.x[0], scale=res.x[1])
            if _sidelobe_energy(f) < _REFINE_BEST:
                return f
    except Exception:
        pass
    return f0


def construct():
    global _REFINE_SHAPE, _REFINE_P0, _REFINE_S0, _REFINE_PMIN, _REFINE_PMAX, _REFINE_SMIN, _REFINE_SMAX, _REFINE_BEST

    # The raw value is what is scored (lower better). Amplitude is free, so we
    # normalize to unit L2 norm of the centered window for a fair, stable metric.
    scale_grid = [1.0]

    candidates = []

    # 1) Raised-cosine family (Hann/Hamming) - best sidelobe suppression
    candidates.append(('hann', np.array([0.0, 0.25, 0.5, 0.75, 1.0])))
    # 2) Gaussian family
    candidates.append(('gaussian', np.array([0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25])))
    # 3) Kaiser family
    candidates.append(('kaiser', np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])))
    # 4) Flat-top family
    candidates.append(('flat_top', np.array([0.05, 0.1, 0.2, 0.3])))

    best_val = np.inf
    best_f = np.ones(N)   # seed fallback

    for shape, grid in candidates:
        if _remaining() < 3.0:
            break
        val, f = _optimize_shape(shape, grid, scale_grid)
        if f is not None and val < best_val:
            best_val = val
            best_f = f

    # Local refinement on the best shape found
    if best_val < np.inf and _remaining() > 15.0:
        # pick the shape that gave best_val by re-deriving; refine hann (usually best)
        _REFINE_SHAPE = 'hann'
        _REFINE_P0 = 1.0
        _REFINE_S0 = 1.0
        _REFINE_PMIN = 0.0
        _REFINE_PMAX = 1.5
        _REFINE_SMIN = 0.1
        _REFINE_SMAX = 5.0
        _REFINE_BEST = best_val
        refined = _local_refine(best_f)
        rv = _sidelobe_energy(refined)
        if rv < best_val:
            best_val = rv
            best_f = refined

    # Normalize to unit L2 norm for a stable, fair raw value
    norm = np.linalg.norm(best_f)
    if norm > 0:
        best_f = best_f / norm

    return best_f


# module-level aliases for refine closure (set inside construct)
_REFINE_SHAPE = 'hann'
_REFINE_P0 = 1.0
_REFINE_S0 = 1.0
_REFINE_PMIN = 0.0
_REFINE_PMAX = 1.5
_REFINE_SMIN = 0.1
_REFINE_SMAX = 5.0
_REFINE_BEST = np.inf
