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

def _circular_convolve(h, n):
    """Circular convolution of h with itself, length n (n >= 2N-1 for no aliasing)."""
    f = np.fft.rfft(h, n=n)
    g = np.fft.irfft(f * f, n=n)
    return g

def _spectral_peak_ratio(mag, n):
    """
    Given nonnegative DFT magnitudes mag (length n//2+1 for rfft),
    compute peak circular autocorrelation / mean circular autocorrelation.
    For zero-phase, circular autocorr = IFFT(|H|^2).
    Mean of circular autocorr = (sum |H|^2) / n = ||h||_2^2 (by Parseval for circular).
    We want to minimize peak / mean.
    """
    H = mag.astype(complex)
    # Reconstruct full spectrum with zero phase
    # rfft gives H[0], H[1..n//2]
    # Full: H_full[0] = mag[0], H_full[k] = mag[k] for k=1..n//2-1, H_full[n//2] = mag[n//2]
    # H_full[n-k] = conj(H_full[k]) = mag[k] (since real and zero phase)
    autocorr = np.fft.irfft(mag * mag, n=n)
    peak = np.max(autocorr)
    mean_ac = np.sum(mag * mag) / n
    if mean_ac <= 0:
        return 0.0, peak, mean_ac
    return peak / mean_ac, peak, mean_ac

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
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Set FFT size: next power of 2 >= 2N-1
    _FFT_SIZE = 1
    while _FFT_SIZE < _CONV_LEN:
        _FFT_SIZE *= 2
    FFT_N = _FFT_SIZE  # use for both linear conv and spectral shaping

    # Coordinates x_i:
    # The intervals are equal width on [-1/4, 1/4].
    # Width w = 0.5 / 600 = 1/1200.
    # Center of interval i: x_i = -0.25 + (i + 0.5) * w.
    w = 0.5 / N
    x = np.linspace(-0.25 + w/2, 0.25 - w/2, N)
    t = x / 0.25  # Map to [-1, 1]

    # --- Spectral shaping: optimize nonnegative DFT magnitudes with zero phase ---
    # For a real, even (zero-phase) signal, the DFT magnitudes are nonnegative.
    # The circular autocorrelation is IFFT(|H|^2).
    # We minimize peak(autocorr) / mean(autocorr) = peak / (||h||_2^2).
    # But our score uses linear conv and (sum h)^2 in denominator.
    # For zero-padding to FFT_N >= 2N-1, the linear conv equals the first N of the circular conv
    # of the zero-padded signal. The peak of linear conv <= peak of circular conv of zero-padded.
    # Actually, for zero-padded signal h_padded (length FFT_N, first N are h, rest 0),
    # circular conv of h_padded with itself equals linear conv of h with itself (padded to FFT_N).
    # So peak of circular autocorr of h_padded = peak of linear autocorr of h.
    # mean of circular autocorr = ||h_padded||_2^2 = ||h||_2^2.
    # But our score denominator is (sum h)^2, not ||h||_2^2.
    # So minimizing peak/||h||_2^2 is not exactly minimizing our score.
    # However, it's a good proxy. We can also directly optimize h in the time domain.
    # Let's do both: spectral shaping to get a good h, then time-domain polish.

    def mag_to_h(mag):
        """Convert nonnegative rfft magnitudes (zero phase) to time-domain h (first N bins)."""
        h_full = np.fft.irfft(mag * mag, n=FFT_N)
        # Wait, that's the autocorrelation, not h.
        # We need h such that |rfft(h)| = mag.
        # For zero phase: h = irfft(mag) ... but mag is the magnitude, and for real even signal,
        # the rfft is real and nonnegative (if h is even and nonnegative? No, h can be anything real).
        # Actually, if we set H = mag (real, nonnegative), then h = irfft(H) is real and even.
        # But h may have negative values. We need h >= 0.
        # So we need to find nonnegative mag such that irfft(mag) >= 0.
        # This is a constraint. Alternatively, we can optimize h directly in time domain
        # using the spectral insight as initialization.
        # Let's just optimize h in time domain with projected gradient, initialized from
        # a spectral shape.
        h = np.fft.irfft(mag, n=FFT_N)
        h = h[:N]
        return h

    def h_to_mag(h):
        """Get nonnegative rfft magnitudes of h (zero-padded to FFT_N)."""
        H = np.fft.rfft(h, n=FFT_N)
        # For zero-phase assumption, we take |H| and assume phase 0.
        # But this only works if h is even. Let's enforce evenness.
        return np.abs(H)

    # Strategy: 
    # 1. Initialize h from a parametric family (as before).
    # 2. Use spectral shaping: set h_even = (h + flip(h))/2, get mag = |rfft(h_even)|,
    #    then optimize mag to minimize peak(iffft(mag^2)) / mean(iffft(mag^2)),
    #    with constraint that irfft(mag) >= 0 (approximately).
    # 3. Project back to h, ensure nonnegativity.
    # 4. Polish in time domain.

    # Simpler: directly optimize h in time domain using the fact that for optimal h,
    # the autocorrelation should be as flat as possible. Use gradient descent on h.

    # --- Family 1: Gaussian-like exp(-(|x|/sigma)^p) ---
    def param_h1(sigma, p):
        if sigma <= 0 or p <= 0:
            return np.ones(N) / N
        h = np.exp(-np.abs(x / sigma) ** p)
        s = np.sum(h)
        if s > 0:
            h = h / s
        return h

    # --- Spectral shaping: optimize DFT magnitudes with zero phase ---
    # Key insight: for a real, even signal h, the DFT is real and nonnegative (zero phase).
    # The autocorrelation is IFFT(|H|^2) = IFFT(H^2) for real nonnegative H.
    # We want to minimize peak(autocorr) subject to sum(h) = 1 and h >= 0.
    # In spectral domain: h = IFFT(H), autocorr = IFFT(H^2).
    # sum(h) = H[0] (DC component) for the rfft convention? 
    # Actually, sum(h) = H[0] * N / FFT_N? Let's be careful.
    # np.fft.rfft(h, n=FFT_N): H[0] = sum(h_padded) = sum(h) (since padded with 0).
    # IFFT(H) = h_padded, so h = h_padded[:N].
    # autocorr = IFFT(H * H, n=FFT_N) (circular conv of h_padded with itself).
    # peak(autocorr) / (sum(h))^2 = peak(autorr) / H[0]^2.
    # We want to minimize this. Let's optimize H directly (nonnegative, real).
    # Constraint: h = IFFT(H)[:N] >= 0.
    # This is a nonconvex constraint, but we can use projected gradient or penalty.

    def spectral_objective(log_mag, beta=200.0):
        """
        Optimize log(mag) to minimize smooth peak of autocorr / H[0]^2.
        log_mag: array of length FFT_N//2 + 1
        """
        mag = np.exp(log_mag)
        # Ensure DC component is reasonable
        H = mag.astype(complex)
        autocorr = np.fft.irfft(mag * mag, n=FFT_N)
        peak = np.max(autocorr)
        H0 = mag[0]
        if H0 <= 0:
            return 1e10
        # Smooth max
        m = peak
        if m <= 0:
            return 0.0
        lse = m + np.log(np.sum(np.exp((autocorr - m) * beta)) / beta)
        return lse / (H0 * H0)

    def spectral_grad(log_mag, beta=200.0):
        """Gradient of spectral_objective w.r.t. log_mag."""
        mag = np.exp(log_mag)
        autocorr = np.fft.irfft(mag * mag, n=FFT_N)
        H0 = mag[0]
        if H0 <= 0:
            return np.zeros_like(log_mag)
        m = np.max(autocorr)
        if m <= 0:
            return np.zeros_like(log_mag)
        w = np.exp((autocorr - m) * beta)
        w /= np.sum(w)
        # d(autocorr[k])/d(mag[j]) = 2 * mag[j] * delta_{k, j} (in circular sense)
        # Actually, autocorr = IFFT(mag^2), so d(autocorr)/d(mag[j]) = 2 * mag[j] * IFFT(e_j)
        # where e_j is the j-th basis vector in freq domain.
        # This is complex. Use finite differences or just do a few steps.
        # For simplicity, use finite differences for the gradient.
        grad = np.zeros_like(log_mag)
        eps = 1e-5
        f0 = spectral_objective(log_mag, beta)
        for i in range(len(log_mag)):
            log_mag_p = log_mag.copy()
            log_mag_p[i] += eps
            fp = spectral_objective(log_mag_p, beta)
            grad[i] = (fp - f0) / eps
        return grad

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

    # --- Spectral shaping optimization ---
    # Initialize from a good parametric family
    # Use a Gaussian-like shape as initial spectral shape
    init_sigma = 0.1
    init_p = 2.0
    h_init = param_h1(init_sigma, init_p)
    # Make it even
    h_even = (h_init + h_init[::-1]) / 2.0
    h_even = np.maximum(0, h_even)
    s_init = np.sum(h_even)
    if s_init > 0:
        h_even = h_even / s_init

    # Get initial magnitudes
    H_init = np.fft.rfft(h_even, n=FFT_N)
    mag_init = np.abs(H_init)
    # Ensure positive
    mag_init = np.maximum(mag_init, 1e-10)
    log_mag_init = np.log(mag_init)

    # Optimize log_mag using L-BFGS-B
    # But the constraint h >= 0 is hard. Let's use a penalty approach.
    def spectral_obj_penalty(log_mag, beta=200.0, lam=100.0):
        mag = np.exp(log_mag)
        H = mag.astype(complex)
        autocorr = np.fft.irfft(mag * mag, n=FFT_N)
        H0 = mag[0]
        if H0 <= 0:
            return 1e10
        m = np.max(autocorr)
        if m <= 0:
            return 0.0
        lse = m + np.log(np.sum(np.exp((autocorr - m) * beta)) / beta)
        score = lse / (H0 * H0)
        # Penalty for negative h
        h_full = np.fft.irfft(mag, n=FFT_N)
        h = h_full[:N]
        neg_penalty = lam * np.sum(np.minimum(0, h) ** 2)
        # Penalty for non-evenness (encourage evenness)
        h_odd = h_full[:N] - h_full[N:FFT_N-N][::-1]
        even_penalty = 0.1 * np.sum(h_odd ** 2)
        return score + neg_penalty + even_penalty

    # Run L-BFGS-B on log_mag
    if time.time() < deadline - 30:
        try:
            res_spec = optimize.minimize(
                spectral_obj_penalty, log_mag_init,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res_spec.fun < 1e10:
                mag_opt = np.exp(res_spec.x)
                h_spec = np.fft.irfft(mag_opt, n=FFT_N)[:N]
                h_spec = np.maximum(0, h_spec)
                s_spec = np.sum(h_spec)
                if s_spec > 0:
                    h_spec = h_spec / s_spec
                    sc_spec = compute_score(h_spec)
                    try_update(h_spec, sc_spec)
        except Exception:
            pass

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

    # Final time-domain polish: projected gradient descent on h
    # Minimize max(conv(h)) / (sum(h))^2
    # Use smooth surrogate for differentiability
    if time.time() < deadline - 10:
        h_polish = best_h.copy()
        # Normalize
        s_polish = np.sum(h_polish)
        if s_polish > 0:
            h_polish = h_polish / s_polish

        # Projected gradient descent with adaptive step size
        beta_pg = 500.0
        lr = 0.01
        for step in range(200):
            if time.time() > deadline - 5:
                break
            # Compute smooth objective and gradient
            s = np.sum(h_polish)
            if s <= 0:
                break
            conv = _fft_convolve(h_polish)
            m = np.max(conv)
            if m <= 0:
                break
            w = np.exp((conv - m) * beta_pg)
            w /= np.sum(w)
            # Gradient of lse w.r.t. conv: w
            # conv[k] = sum_i h[i] * h[k-i] (linear conv, k from 0 to 2N-2)
            # d(conv[k])/d(h[j]) = h[k-j] if 0 <= k-j < N, else 0
            #   + h[j-(k-N+1)] if ... this is the linear conv gradient
            # Actually, for linear conv: conv[k] = sum_{i} h[i] * h[k-i] where h is defined on [0,N-1]
            # and h[k-i] = 0 if k-i not in [0,N-1].
            # d(conv[k])/d(h[j]) = h[k-j] + h[j-(k-N+1)] ... no.
            # Let's use: conv = h * h (linear). The gradient of sum_k w_k conv[k] w.r.t. h[j]
            # = sum_k w_k (h[k-j] + h[j-k+N-1]) ... this is getting complex.
            # Simpler: use finite difference gradient for a few steps.
            # Or just use Nelder-Mead on a small set of parameters.
            break  # Skip gradient descent, rely on parametric optimization

    # Final check: ensure best_h is valid (non-negative)
    best_h = np.maximum(0, best_h)

    return best_h.tolist()
# EVOLVE-BLOCK-END
