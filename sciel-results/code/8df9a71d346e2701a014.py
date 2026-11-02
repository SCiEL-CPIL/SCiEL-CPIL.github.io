"""Program using model-and-optimize approach for first autocorrelation inequality."""
import time
import numpy as np
from scipy import optimize

N = 600
_CONV_LEN = 2 * N - 1
_FFT_SIZE = 1  # set in construct()

def _fft_convolve(h):
    """FFT-based linear convolution (full mode)."""
    n = _FFT_SIZE
    f = np.fft.rfft(h, n=n)
    g = np.fft.irfft(f * f, n=n)[:_CONV_LEN]
    return g

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = _fft_convolve(h)
    max_conv = np.max(conv)
    return max_conv / (s * s)

def compute_score_smooth(h, beta=500.0):
    """
    Smooth surrogate: log-sum-exp based approximation of max(conv) / s^2.
    More differentiable than hard max, helps gradient-based optimizers.
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = _fft_convolve(h)
    # log-sum-exp for numerical stability
    m = np.max(conv)
    if m <= 0:
        return 0.0
    lse = m + np.log(np.sum(np.exp((conv - m) * beta)) / beta)
    return lse / (s * s)

# EVOLVE-BLOCK-START
def construct():
    global _FFT_SIZE
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    # Set FFT size: next power of 2 >= 2N-1
    _FFT_SIZE = 1
    while _FFT_SIZE < _CONV_LEN:
        _FFT_SIZE *= 2

    # Coordinates x_i:
    # The intervals are equal width on [-1/4, 1/4].
    # Width w = 0.5 / 600 = 1/1200.
    # Center of interval i: x_i = -0.25 + (i + 0.5) * w.
    w = 0.5 / N
    x = np.linspace(-0.25 + w/2, 0.25 - w/2, N)

    # --- Family 1: Gaussian-like exp(-(|x|/sigma)^p) ---
    def param_h1(sigma, p):
        if sigma <= 0 or p <= 0:
            return np.ones(N) / N
        h = np.exp(-np.abs(x / sigma) ** p)
        s = np.sum(h)
        if s > 0:
            h = h / s
        return h

    # --- Family 2: Polynomial bump (1 - |t|^a)^b with support [-1,1] in t-space ---
    def param_h2(a, b):
        if a <= 0 or b <= 0:
            return np.ones(N) / N
        t = x / 0.25
        abs_t = np.abs(t)
        h = np.where(abs_t < 1.0, (1.0 - abs_t ** a) ** b, 0.0)
        s = np.sum(h)
        if s > 0:
            h = h / s
        return h

    # --- Family 3: exp(-(|x|/sigma)^p) with adjustable support cutoff ---
    # h(x) = exp(-(|x|/sigma)^p) for |x| < L, 0 otherwise
    # Parameters: sigma (scale), p (shape), L (half-support in x-units, max 0.25)
    def param_h3(sigma, p, L):
        if sigma <= 0 or p <= 0 or L <= 0:
            return np.ones(N) / N
        L = min(L, 0.25)
        h = np.exp(-np.abs(x / sigma) ** p)
        h[np.abs(x) >= L] = 0.0
        s = np.sum(h)
        if s > 0:
            h = h / s
        return h

    # Objective functions using smooth surrogate for optimization
    def obj1(params, beta=500.0):
        sigma, p = params
        sigma = np.clip(sigma, 0.005, 0.5)
        p = np.clip(p, 0.3, 15.0)
        h = param_h1(sigma, p)
        return compute_score_smooth(h, beta)

    def obj2(params, beta=500.0):
        a, b = params
        a = np.clip(a, 0.1, 15.0)
        b = np.clip(b, 0.1, 15.0)
        h = param_h2(a, b)
        return compute_score_smooth(h, beta)

    def obj3(params, beta=500.0):
        sigma, p, L = params
        sigma = np.clip(sigma, 0.005, 0.5)
        p = np.clip(p, 0.3, 15.0)
        L = np.clip(L, 0.02, 0.25)
        h = param_h3(sigma, p, L)
        return compute_score_smooth(h, beta)

    # Track best
    best_h = np.ones(N) / N
    best_score = compute_score(best_h)

    def try_update(h, score):
        nonlocal best_h, best_score
        if score < best_score:
            best_score = score
            best_h = h.copy()

    # Grid search over all three families
    grid1 = [(s, p) for s in [0.03, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2] for p in [0.5, 1, 1.5, 2, 3, 4, 6]]
    grid2 = [(a, b) for a in [0.5, 1, 2, 3, 4, 6, 8] for b in [0.5, 1, 2, 3, 4, 6, 8]]
    grid3 = [(s, p, L) for s in [0.03, 0.05, 0.08, 0.1, 0.15] for p in [1, 2, 3, 4] for L in [0.1, 0.15, 0.2, 0.25]]

    for s0, p0 in grid1:
        if time.time() > deadline:
            break
        h0 = param_h1(s0, p0)
        sc = compute_score(h0)
        try_update(h0, sc)

    for a0, b0 in grid2:
        if time.time() > deadline:
            break
        h0 = param_h2(a0, b0)
        sc = compute_score(h0)
        try_update(h0, sc)

    for s0, p0, L0 in grid3:
        if time.time() > deadline:
            break
        h0 = param_h3(s0, p0, L0)
        sc = compute_score(h0)
        try_update(h0, sc)

    # Local optimization with smooth surrogate, multiple restarts
    # Family 1
    for s0, p0 in grid1:
        if time.time() > deadline:
            break
        bounds = [(0.005, 0.5), (0.3, 15.0)]
        try:
            res = optimize.minimize(
                obj1, [s0, p0], method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 80, 'ftol': 1e-10}
            )
            if res.fun < best_score:
                h_cand = param_h1(*np.clip(res.x, [0.005, 0.3], [0.5, 15.0]))
                sc = compute_score(h_cand)
                try_update(h_cand, sc)
        except Exception:
            pass

    # Family 2
    for a0, b0 in grid2:
        if time.time() > deadline:
            break
        bounds = [(0.1, 15.0), (0.1, 15.0)]
        try:
            res = optimize.minimize(
                obj2, [a0, b0], method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 80, 'ftol': 1e-10}
            )
            if res.fun < best_score:
                h_cand = param_h2(*np.clip(res.x, [0.1, 0.1], [15.0, 15.0]))
                sc = compute_score(h_cand)
                try_update(h_cand, sc)
        except Exception:
            pass

    # Family 3
    for s0, p0, L0 in grid3:
        if time.time() > deadline:
            break
        bounds = [(0.005, 0.5), (0.3, 15.0), (0.02, 0.25)]
        try:
            res = optimize.minimize(
                obj3, [s0, p0, L0], method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 80, 'ftol': 1e-10}
            )
            if res.fun < best_score:
                x_c = np.clip(res.x, [0.005, 0.3, 0.02], [0.5, 15.0, 0.25])
                h_cand = param_h3(*x_c)
                sc = compute_score(h_cand)
                try_update(h_cand, sc)
        except Exception:
            pass

    # Final polish: if we have time, do a tighter optimization from the best found
    # Re-evaluate best_score exactly (not via surrogate)
    best_score = compute_score(best_h)

    # If time remains, try a few more targeted optimizations from the best region
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        # Try COBYLA from best point for family 1 (if best came from family 1)
        # We don't track which family, so just do a couple extra L-BFGS runs
        extra_starts1 = [(0.05, 1.0), (0.08, 2.0), (0.1, 3.0), (0.06, 1.5), (0.12, 2.5)]
        for s0, p0 in extra_starts1:
            if time.time() > deadline:
                break
            bounds = [(0.005, 0.5), (0.3, 15.0)]
            try:
                res = optimize.minimize(
                    obj1, [s0, p0], method='L-BFGS-B',
                    bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12}
                )
                if res.fun < best_score:
                    h_cand = param_h1(*np.clip(res.x, [0.005, 0.3], [0.5, 15.0]))
                    sc = compute_score(h_cand)
                    try_update(h_cand, sc)
            except Exception:
                pass

        extra_starts2 = [(1.0, 2.0), (2.0, 3.0), (0.5, 1.0), (3.0, 2.0)]
        for a0, b0 in extra_starts2:
            if time.time() > deadline:
                break
            bounds = [(0.1, 15.0), (0.1, 15.0)]
            try:
                res = optimize.minimize(
                    obj2, [a0, b0], method='L-BFGS-B',
                    bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12}
                )
                if res.fun < best_score:
                    h_cand = param_h2(*np.clip(res.x, [0.1, 0.1], [15.0, 15.0]))
                    sc = compute_score(h_cand)
                    try_update(h_cand, sc)
            except Exception:
                pass

        extra_starts3 = [(0.05, 2.0, 0.15), (0.08, 3.0, 0.2), (0.06, 1.5, 0.12), (0.1, 2.0, 0.25)]
        for s0, p0, L0 in extra_starts3:
            if time.time() > deadline:
                break
            bounds = [(0.005, 0.5), (0.3, 15.0), (0.02, 0.25)]
            try:
                res = optimize.minimize(
                    obj3, [s0, p0, L0], method='L-BFGS-B',
                    bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12}
                )
                if res.fun < best_score:
                    x_c = np.clip(res.x, [0.005, 0.3, 0.02], [0.5, 15.0, 0.25])
                    h_cand = param_h3(*x_c)
                    sc = compute_score(h_cand)
                    try_update(h_cand, sc)
            except Exception:
                pass

    # Final check: ensure best_h is valid (non-negative)
    best_h = np.maximum(0, best_h)

    return best_h.tolist()
# EVOLVE-BLOCK-END
