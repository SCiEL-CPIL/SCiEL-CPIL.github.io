"""Combined: band-limited spectral parameterization + fast FFT eval + full-height polish."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 600
DEADLINE = 142.0  # safety margin before the 150s wall-clock kill

# --- Band-limited model parameters ---
BAND = 30                 # number of cosine modes
FALL = 0.85               # geometric falloff of the basis (keeps signal smooth)
LAMBDA = 0.02             # L2 penalty toward the band-limited family during polish
BETA_RANGE = (0.5, 4.0)   # tunable "sharpness" of the spectral profile

_dx = 1.0 / (2.0 * N)


def _autoconv_max_normalized(heights):
    """Normalized maximum of the autoconvolution of the step function."""
    h = np.asarray(heights, dtype=np.float64)
    n = len(h)
    padded = np.zeros(2 * n, dtype=np.float64)
    padded[:n] = h
    F = np.fft.fft(padded)
    G = F * np.conj(F)
    conv = np.fft.ifft(G).real
    max_conv = np.max(conv[:2 * n - 1])
    total = np.sum(h)
    if total == 0:
        return 0.0
    dx = 1.0 / (2.0 * n)
    return max_conv / (total * total * dx)


def _compute_ratio_fast(h):
    """Fast normalized max autoconvolution ratio (FFT based)."""
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    padded = np.zeros(2 * n, dtype=np.float64)
    padded[:n] = h
    F = np.fft.fft(padded)
    conv = np.fft.ifft(F * np.conj(F)).real
    max_conv = np.max(conv[:2 * n - 1])
    total = np.sum(h)
    if total == 0:
        return float('inf')
    dx = 1.0 / (2.0 * n)
    return max_conv / (total * total * dx)


# Precompute the band-limited basis matrix (rows = sample index, cols = [beta, c0..cB])
_idx = np.arange(N, dtype=np.float64)
_cos_terms = np.cos(2.0 * np.pi * _idx * np.arange(BAND + 1) / N)  # (N, BAND+1)
_basis = np.hstack([np.ones((N, 1)), _cos_terms[:, 1:]])           # (N, BAND+1)
_basis_scale = np.sum(_basis[:, 1:] * _basis[:, 1:], axis=0)       # for normalization


def _parametric_heights(params):
    """
    Build a smooth, nonnegative, band-limited step function.
    params = [beta, c1, ..., cB]
    beta (sharpness) shapes the spectral profile:
        a0 = 1.0,  a_k = exp(-beta * (k/(B+1))**2)  (k>=1)
    c_k are small shape perturbations.  The result is scaled so the mean height is 1.
    """
    beta = float(params[0])
    c = params[1:]
    k = np.arange(BAND + 1)
    a = np.exp(-beta * (k / (BAND + 1.0)) ** 2)
    a[0] = 1.0
    # add shape perturbations for k>=1
    a[1:] += c * FALL ** k
    h = _basis_scale[0] * a[0] + _basis_scale[1:] @ a[1:]  # sum of h over samples
    if h <= 0:
        h = np.ones(N)
        return h
    hvals = _basis @ a
    hvals = np.maximum(hvals, 0.0)
    s = np.sum(hvals)
    if s > 0:
        hvals = hvals / s * N
    return hvals


def _objective_parametric(params):
    h = _parametric_heights(params)
    return _compute_ratio_fast(h)


def _bandlimited_reference(beta):
    """Pure band-limited reference (no perturbations) for the L2 pull."""
    k = np.arange(BAND + 1)
    a = np.exp(-beta * (k / (BAND + 1.0)) ** 2)
    a[0] = 1.0
    h = _basis @ a
    h = np.maximum(h, 0.0)
    s = np.sum(h)
    if s > 0:
        h = h / s * N
    return h


def construct():
    start_time = time.time()

    def _elapsed():
        return time.time() - start_time

    # ---- Baseline: uniform ----
    best_heights = np.ones(N, dtype=np.float64)
    best_score = _compute_ratio_fast(best_heights)

    # ---- Sweep a few sharpness values for cheap initial candidates ----
    for beta in [0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 3.5]:
        h = _parametric_heights(np.concatenate(([beta], np.zeros(BAND))))
        sc = _compute_ratio_fast(h)
        if sc < best_score:
            best_score = sc
            best_heights = h.copy()

    # ---- Global search over the band-limited parameter space ----
    if _elapsed() < DEADLINE - 40.0:
        try:
            bounds = [BETA_RANGE] + [(-0.4, 0.4)] * BAND
            result_de = differential_evolution(
                _objective_parametric,
                bounds=bounds,
                seed=7,
                maxiter=40,
                popsize=12,
                tol=1e-7,
                polish=True,
                init='sobol',
            )
            if result_de.fun < best_score:
                h_opt = _parametric_heights(result_de.x)
                h_opt = np.maximum(h_opt, 0.0)
                sc = _compute_ratio_fast(h_opt)
                if sc < best_score:
                    best_score = sc
                    best_heights = h_opt.copy()
        except Exception:
            pass

    # ---- Local polish in full height space with a pull toward the band family ----
    if _elapsed() < DEADLINE - 12.0:
        try:
            h0 = best_heights.copy()
            # choose a reference sharpness close to the current shape (use a mid value)
            ref = _bandlimited_reference(1.3)

            def _smooth_objective(h):
                hp = np.maximum(h, 0.0)
                sc = _compute_ratio_fast(hp)
                if not np.isfinite(sc):
                    return 1e10
                diff = np.diff(hp)
                penalty = 1e-4 * np.sum(diff * diff) + LAMBDA * np.sum((hp - ref) ** 2)
                return sc + penalty

            result_local = minimize(
                _smooth_objective,
                h0,
                method='L-BFGS-B',
                bounds=[(0.0, None)] * N,
                options={'maxiter': 120, 'ftol': 1e-12, 'gtol': 1e-6},
            )
            if result_local.fun < best_score + 1e-6:
                h_opt = np.maximum(result_local.x, 0.0)
                sc = _compute_ratio_fast(h_opt)
                if sc < best_score:
                    best_score = sc
                    best_heights = h_opt.copy()
        except Exception:
            pass

    return best_heights.tolist()
