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
    t = x / 0.25  # Map to [-1, 1]

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
        abs_t = np.abs(t)
        h = np.where(abs_t < 1.0, (1.0 - abs_t ** a) ** b, 0.0)
        s = np.sum(h)
        if s > 0:
            h = h / s
        return h

    # --- Family 3: exp(-(|x|/sigma)^p) with adjustable support cutoff ---
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

    # --- Family 4: Flat-topped bump with adjustable plateau width ---
    # h(t) = 1 for |t| < c, (1-|t|^a)^b for c <= |t| < 1, 0 for |t| >= 1
    def param_h4(c, a, b):
        if c < 0 or c >= 1 or a <= 0 or b <= 0:
            return np.ones(N) / N
        abs_t = np.abs(t)
        h = np.zeros(N)
        mask_plateau = abs_t < c
        mask_transition = (abs_t >= c) & (abs_t < 1.0)
        h[mask_plateau] = 1.0
        if np.any(mask_transition):
            # Smooth transition: scale so that at t=c, value matches 1
            # (1 - (t/c)^a)^b doesn't work directly; use (1 - ((t-c)/(1-c))^a)^b
            tc = (abs_t[mask_transition] - c) / (1.0 - c)
            h[mask_transition] = (1.0 - tc ** a) ** b
        s = np.sum(h)
        if s > 0:
            h = h / s
        return h

    # --- Family 5: Two-sided polynomial bump on full support [-1,1] ---
    # h(t) = (1 - |t|^a)^b for |t| < 1, 0 otherwise
    # This is the natural candidate: symmetric, full support, smooth
    def param_h5(a, b):
        if a <= 0 or b <= 0:
            return np.ones(N) / N
        abs_t = np.abs(t)
        h = np.where(abs_t < 1.0, (1.0 - abs_t ** a) ** b, 0.0)
        s = np.sum(h)
        if s > 0:
            h = h / s
        return h

    # --- Family 6: Cosine bump on full support ---
    # h(t) = cos(pi * |t| / 2)^p for |t| < 1, 0 otherwise
    def param_h6(p):
        if p <= 0:
            return np.ones(N) / N
        abs_t = np.abs(t)
        h = np.where(abs_t < 1.0, np.cos(np.pi * abs_t / 2.0) ** p, 0.0)
        s = np.sum(h)
        if s > 0:
            h = h / s
        return h

    # Objective functions using smooth surrogate for optimization
    def obj1(params, beta=800.0):
        sigma, p = params
        sigma = np.clip(sigma, 0.005, 0.5)
        p = np.clip(p, 0.3, 15.0)
        h = param_h1(sigma, p)
        return compute_score_smooth(h, beta)

    def obj2(params, beta=800.0):
        a, b = params
        a = np.clip(a, 0.1, 15.0)
        b = np.clip(b, 0.1, 15.0)
        h = param_h2(a, b)
        return compute_score_smooth(h, beta)

    def obj3(params, beta=800.0):
        sigma, p, L = params
        sigma = np.clip(sigma, 0.005, 0.5)
        p = np.clip(p, 0.3, 15.0)
        L = np.clip(L, 0.02, 0.25)
        h = param_h3(sigma, p, L)
        return compute_score_smooth(h, beta)

    def obj4(params, beta=800.0):
        c, a, b = params
        c = np.clip(c, 0.0, 0.95)
        a = np.clip(a, 0.5, 10.0)
        b = np.clip(b, 0.5, 10.0)
        h = param_h4(c, a, b)
        return compute_score_smooth(h, beta)

    def obj5(params, beta=800.0):
        a, b = params
        a = np.clip(a, 0.1, 20.0)
        b = np.clip(b, 0.1, 20.0)
        h = param_h5(a, b)
        return compute_score_smooth(h, beta)

    def obj6(params, beta=800.0):
        p = params[0]
        p = np.clip(p, 0.1, 20.0)
        h = param_h6(p)
        return compute_score_smooth(h, beta)

    # Track best
    best_h = np.ones(N) / N
    best_score = compute_score(best_h)

    def try_update(h, score):
        nonlocal best_h, best_score
        if score < best_score:
            best_score = score
            best_h = h.copy()

    # Grid search over all four families
    grid1 = [(s, p) for s in [0.03, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2] for p in [0.5, 1, 1.5, 2, 3, 4, 6]]
    grid2 = [(a, b) for a in [0.5, 1, 2, 3, 4, 6, 8] for b in [0.5, 1, 2, 3, 4, 6, 8]]
    grid3 = [(s, p, L) for s in [0.03, 0.05, 0.08, 0.1, 0.15] for p in [1, 2, 3, 4] for L in [0.1, 0.15, 0.2, 0.25]]
    grid4 = [(c, a, b) for c in [0.0, 0.2, 0.4, 0.6] for a in [1, 2, 4] for b in [1, 2, 4]]
    grid5 = [(a, b) for a in [0.5, 1, 2, 3, 4, 6, 8, 12] for b in [0.5, 1, 2, 3, 4, 6, 8, 12]]
    grid6 = [p for p in [0.5, 1, 2, 3, 4, 6, 8, 12]]

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

    for c0, a0, b0 in grid4:
        if time.time() > deadline:
            break
        h0 = param_h4(c0, a0, b0)
        sc = compute_score(h0)
        try_update(h0, sc)

    for a0, b0 in grid5:
        if time.time() > deadline:
            break
        h0 = param_h5(a0, b0)
        sc = compute_score(h0)
        try_update(h0, sc)

    for p0 in grid6:
        if time.time() > deadline:
            break
        h0 = param_h6(p0)
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
                bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12}
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
                bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12}
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
                bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12}
            )
            if res.fun < best_score:
                x_c = np.clip(res.x, [0.005, 0.3, 0.02], [0.5, 15.0, 0.25])
                h_cand = param_h3(*x_c)
                sc = compute_score(h_cand)
                try_update(h_cand, sc)
        except Exception:
            pass

    # Family 4
    for c0, a0, b0 in grid4:
        if time.time() > deadline:
            break
        bounds = [(0.0, 0.95), (0.5, 10.0), (0.5, 10.0)]
        try:
            res = optimize.minimize(
                obj4, [c0, a0, b0], method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12}
            )
            if res.fun < best_score:
                x_c = np.clip(res.x, [0.0, 0.5, 0.5], [0.95, 10.0, 10.0])
                h_cand = param_h4(*x_c)
                sc = compute_score(h_cand)
                try_update(h_cand, sc)
        except Exception:
            pass

    # Family 5: Two-sided polynomial bump
    for a0, b0 in grid5:
        if time.time() > deadline:
            break
        bounds = [(0.1, 20.0), (0.1, 20.0)]
        try:
            res = optimize.minimize(
                obj5, [a0, b0], method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12}
            )
            if res.fun < best_score:
                x_c = np.clip(res.x, [0.1, 0.1], [20.0, 20.0])
                h_cand = param_h5(*x_c)
                sc = compute_score(h_cand)
                try_update(h_cand, sc)
        except Exception:
            pass

    # Family 6: Cosine bump
    for p0 in grid6:
        if time.time() > deadline:
            break
        bounds = [(0.1, 20.0)]
        try:
            res = optimize.minimize(
                obj6, [p0], method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12}
            )
            if res.fun < best_score:
                x_c = np.clip(res.x, [0.1], [20.0])
                h_cand = param_h6(x_c[0])
                sc = compute_score(h_cand)
                try_update(h_cand, sc)
        except Exception:
            pass

    # Final polish: direct coordinate optimization using projected gradient descent
    # on the actual 600-dim vector, starting from best_h
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        # Projected gradient descent on h directly
        # Objective: max(conv(h)) / (sum(h))^2
        # Gradient of max(conv) w.r.t. h: conv is quadratic, so grad = 2 * convolve(h, e_k) where e_k is argmax
        # But we want to minimize max(conv)/s^2. Let's use the smooth surrogate gradient.
        
        def grad_smooth_h(h, beta=800.0):
            s = np.sum(h)
            if s == 0:
                return np.zeros(N)
            conv = _fft_convolve(h)
            m = np.max(conv)
            if m <= 0:
                return np.zeros(N)
            # Soft max weights
            w = np.exp((conv - m) * beta)
            w /= np.sum(w)
            # Gradient of log-sum-exp w.r.t. conv elements = w
            # conv[k] = sum_i h[i]*h[k-i] (for linear conv, indices shifted)
            # d(conv[k])/d(h[j]) = h[k-j] + h[j-(k-N+1)] ... this is complex.
            # Instead, use finite differences or just do a few Nelder-Mead steps.
            return None  # Skip gradient, use Nelder-Mead
        
        # Use Nelder-Mead on the parametric representation of the best family
        # But we don't know which family gave best_h. Let's just do a few
        # high-precision L-BFGS runs from the best point for each family.
        
        # Re-evaluate best_score exactly
        best_score = compute_score(best_h)
        
        # Try COBYLA from several good starting points for each family
        extra_starts1 = [(0.05, 1.0), (0.08, 2.0), (0.1, 3.0), (0.06, 1.5), (0.12, 2.5), (0.04, 1.0), (0.07, 2.5)]
        for s0, p0 in extra_starts1:
            if time.time() > deadline - 5:
                break
            bounds = [(0.005, 0.5), (0.3, 15.0)]
            try:
                res = optimize.minimize(
                    obj1, [s0, p0], method='L-BFGS-B',
                    bounds=bounds, options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res.fun < best_score:
                    h_cand = param_h1(*np.clip(res.x, [0.005, 0.3], [0.5, 15.0]))
                    sc = compute_score(h_cand)
                    try_update(h_cand, sc)
            except Exception:
                pass

        extra_starts2 = [(1.0, 2.0), (2.0, 3.0), (0.5, 1.0), (3.0, 2.0), (1.5, 1.5), (4.0, 1.0), (0.8, 2.5)]
        for a0, b0 in extra_starts2:
            if time.time() > deadline - 5:
                break
            bounds = [(0.1, 15.0), (0.1, 15.0)]
            try:
                res = optimize.minimize(
                    obj2, [a0, b0], method='L-BFGS-B',
                    bounds=bounds, options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res.fun < best_score:
                    h_cand = param_h2(*np.clip(res.x, [0.1, 0.1], [15.0, 15.0]))
                    sc = compute_score(h_cand)
                    try_update(h_cand, sc)
            except Exception:
                pass

        extra_starts3 = [(0.05, 2.0, 0.15), (0.08, 3.0, 0.2), (0.06, 1.5, 0.12), (0.1, 2.0, 0.25), (0.04, 2.5, 0.1), (0.07, 1.5, 0.18)]
        for s0, p0, L0 in extra_starts3:
            if time.time() > deadline - 5:
                break
            bounds = [(0.005, 0.5), (0.3, 15.0), (0.02, 0.25)]
            try:
                res = optimize.minimize(
                    obj3, [s0, p0, L0], method='L-BFGS-B',
                    bounds=bounds, options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res.fun < best_score:
                    x_c = np.clip(res.x, [0.005, 0.3, 0.02], [0.5, 15.0, 0.25])
                    h_cand = param_h3(*x_c)
                    sc = compute_score(h_cand)
                    try_update(h_cand, sc)
            except Exception:
                pass

        extra_starts4 = [(0.3, 2.0, 2.0), (0.5, 1.5, 1.5), (0.2, 3.0, 3.0), (0.4, 2.0, 1.0), (0.6, 1.0, 2.0)]
        for c0, a0, b0 in extra_starts4:
            if time.time() > deadline - 5:
                break
            bounds = [(0.0, 0.95), (0.5, 10.0), (0.5, 10.0)]
            try:
                res = optimize.minimize(
                    obj4, [c0, a0, b0], method='L-BFGS-B',
                    bounds=bounds, options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res.fun < best_score:
                    x_c = np.clip(res.x, [0.0, 0.5, 0.5], [0.95, 10.0, 10.0])
                    h_cand = param_h4(*x_c)
                    sc = compute_score(h_cand)
                    try_update(h_cand, sc)
            except Exception:
                pass

    # Final high-precision polish: run L-BFGS with tighter tolerances from best points
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        # Polish family 5 (two-sided polynomial) with tight tolerance
        # Try several good starting points
        polish_starts5 = [(1.0, 1.0), (2.0, 2.0), (0.5, 1.5), (1.5, 1.0), (2.0, 1.5), (1.0, 2.0), (3.0, 2.0), (1.5, 2.0)]
        for a0, b0 in polish_starts5:
            if time.time() > deadline - 5:
                break
            bounds = [(0.1, 20.0), (0.1, 20.0)]
            try:
                res = optimize.minimize(
                    obj5, [a0, b0], method='L-BFGS-B',
                    bounds=bounds, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res.fun < best_score:
                    x_c = np.clip(res.x, [0.1, 0.1], [20.0, 20.0])
                    h_cand = param_h5(*x_c)
                    sc = compute_score(h_cand)
                    try_update(h_cand, sc)
            except Exception:
                pass

        # Polish family 6 (cosine) with tight tolerance
        polish_starts6 = [1.0, 2.0, 1.5, 0.8, 2.5, 1.2, 3.0]
        for p0 in polish_starts6:
            if time.time() > deadline - 5:
                break
            bounds = [(0.1, 20.0)]
            try:
                res = optimize.minimize(
                    obj6, [p0], method='L-BFGS-B',
                    bounds=bounds, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res.fun < best_score:
                    x_c = np.clip(res.x, [0.1], [20.0])
                    h_cand = param_h6(x_c[0])
                    sc = compute_score(h_cand)
                    try_update(h_cand, sc)
            except Exception:
                pass

    # Final check: ensure best_h is valid (non-negative)
    best_h = np.maximum(0, best_h)

    return best_h.tolist()
# EVOLVE-BLOCK-END
