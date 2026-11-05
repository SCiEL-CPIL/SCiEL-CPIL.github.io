"""Model-and-optimize program for third autocorrelation inequality.

Strategy:
  * Represent h as a constant baseline plus a zero-mean oscillatory perturbation
    parameterized by DCT coefficients (low-frequency spectral shaping).
  * Use FFT-based soft-max surrogate with exact analytic gradient via FFT
    correlation (fast O(n log n) per evaluation).
  * Add a spectral shaping phase that exploits the Toeplitz structure of the
    autoconvolution to initialize DCT coefficients that spread peak energy.
  * Polish with L-BFGS-B on full 400-dim vector, then coordinate/subgradient
    refinement on the hard objective.
"""
N = 400
import time
import numpy as np
from scipy.optimize import minimize
from scipy.fftpack import dct, idct


def _autoconv(h):
    """Full autoconvolution via FFT (c[k] for lag k = -(N-1)..(N-1))."""
    n = len(h)
    f = np.fft.rfft(h, n=2 * n)
    c = np.fft.irfft(f * f, n=2 * n)
    return c


def _soft_max_conv(h, beta):
    """Return (soft_max_value, gradient w.r.t. h) using FFT-based gradient."""
    n = len(h)
    conv = _autoconv(h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    ez = np.exp(shifted)
    z = ez.sum()
    sm = m + np.log(z) / beta
    w = ez / z

    # Gradient of soft-max w.r.t. h_i = sum_k w_k * d(conv_k)/d(h_i)
    # d(conv_k)/d(h_i) = h[k-i] + h[i-k] (zero outside 0..n-1)
    # grad_i = sum_k w_k h[k-i] + sum_k w_k h[i-k]
    #
    # Term 2: sum_k w_k h[i-k] = (w conv h)[i] = A[i] where A = convolve(w, h, 'full')
    # Term 1: sum_k w_k h[k-i] = sum_j h_j w[j+i] = cross-correlation of h and w at lag i
    #
    # Use FFT for both terms:
    # For term 2: A = ifft(FFT(w) * FFT(h)) gives convolution, take first n elements
    # For term 1: B = ifft(FFT(h) * conj(FFT(w))) gives cross-correlation
    #   B[p] = sum_k h_k conj(w)_{p-k} = sum_k h_k w[p-k] (real)
    #   We want sum_j h_j w[j+i] = sum_j h_j w[j+i]
    #   = B at index (n-1+i) in the full correlation of length 2n-1
    #   Actually: np.correlate(h, w, 'full')[p] = sum_k h_k w[k+p-(n-1)]
    #   So np.correlate(h, w, 'full')[n-1+i] = sum_k h_k w[k+i] = sum_j h_j w[j+i] ✓

    # Compute via FFT for speed
    W = np.fft.rfft(w, n=2 * n)
    H = np.fft.rfft(h, n=2 * n)

    # Term 2: (w conv h)[i] for i=0..n-1
    conv_wh = np.fft.irfft(W * H, n=2 * n)
    term2 = conv_wh[0:n]

    # Term 1: sum_j h_j w[j+i] for i=0..n-1
    # Cross-correlation: ifft(conj(H) * W) gives sum_k h_k w[i-k]... 
    # Actually let's use: corr = ifft(H * conj(W)) 
    # corr[p] = sum_k h_k w[p-k] (this is convolution of h with flipped w)
    # We want sum_j h_j w[j+i] = sum_j h_j w[(j+i)]
    # Let's use np.correlate approach via FFT:
    # correlate(h, w, 'full')[p] = sum_k h_k w[k + p - (n-1)]
    # We want p = n-1+i: sum_k h_k w[k+i] ✓
    # FFT of correlate: ifft(conj(FFT(h)) * FFT(w)) padded appropriately
    # But simpler: use the fact that for real signals,
    # correlate(h,w)[p] = convolve(h, flip(w))[p + n - 1]
    # flip(w) has FFT = conj(W) (for real w, with proper indexing)
    # Actually, let's just compute directly since n=400 is small enough
    # for one extra O(n^2) pass, but let's try FFT:
    
    # Alternative: term1_i = sum_k w_k h[k-i]
    # = sum_k w_k h[k-i]
    # Let's compute B = convolve(w, h, 'full') = A (same as term2 source)
    # A[p] = sum_k w_k h[p-k]
    # term2_i = A[i] = sum_k w_k h[i-k] ✓
    # term1_i = sum_k w_k h[k-i]
    # Let j = k-i: sum_j w_{j+i} h[j] = sum_j h[j] w[j+i]
    # This is A at a different index? A[p] = sum_k w_k h[p-k]
    # sum_j h[j] w[j+i] = sum_j w[j+i] h[j]
    # Let q = j+i: sum_q w[q] h[q-i] = A[q-i]... no that's not right either.
    # 
    # Direct: sum_j h[j] w[j+i] for fixed i
    # = cross-correlation of h and w at lag i
    # = np.correlate(h, w, 'full')[n-1+i]
    # 
    # Via FFT: correlate(h,w) = ifft(conj(FFT(h_padded)) * FFT(w_padded))
    # where padding to length >= 2n-1
    # For length 2n:
    H2 = np.fft.rfft(h, n=2 * n)
    W2 = np.fft.rfft(w, n=2 * n)
    # corr_full = ifft(conj(H2) * W2) gives correlate(h, w) with proper indexing
    corr_full = np.fft.irfft(np.conj(H2) * W2, n=2 * n)
    # corr_full[p] = sum_k h_k w[p-k] for p in 0..2n-1 (circular, but with zero-padding it's linear)
    # Actually with zero-padding to 2n, this gives linear cross-correlation:
    # corr_full[p] = sum_k h[k] w[p-k] where h and w are zero-padded to length 2n
    # For p in 0..2n-2, this equals the linear cross-correlation
    # correlate(h,w,'full')[q] = sum_k h[k] w[k+q-(n-1)]
    # So corr_full[p] = correlate(h,w,'full')[p] when p >= n-1? 
    # Let me verify: corr_full[p] = sum_k h[k] w[p-k] (with zero padding)
    # correlate(h,w,'full')[q] = sum_k h[k] w[k+q-(n-1)]
    # Set k' = k: corr_full[p] = sum_k h[k] w[p-k]
    # correlate(h,w,'full')[q] = sum_k h[k] w[k+q-n+1]
    # If we set p-k = k+q-n+1 => p = 2k+q-n+1, not a simple shift.
    # 
    # Let me just use np.correlate directly for correctness (n=400, O(n^2)=160k ops):
    B = np.correlate(h, w, mode='full')  # B[p] = sum_k h[k] w[k+p-(n-1)]
    term1 = B[n-1:2*n-1]  # term1[i] = B[n-1+i] = sum_k h[k] w[k+i]

    grad = term1 + term2
    return sm, grad


def _obj_and_grad(h, beta):
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e12, np.zeros_like(h)
    sm, g_sm = _soft_max_conv(h, beta)
    val = 2.0 * N * sm / (s * s)
    g = 2.0 * N * (g_sm * s - 2.0 * sm) / (s * s * s)
    return val, g


def _hard(h):
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e12
    conv = _autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _param_to_h(p):
    """Map parameter vector to a 400-length step-height vector.
    p = [baseline, dct_coeff_1, ..., dct_coeff_m]
    """
    baseline = p[0]
    m = len(p) - 1
    coeffs = np.zeros(N)
    coeffs[0] = 0.0
    coeffs[1:m+1] = p[1:m+1]
    h_osc = idct(coeffs, type=2, norm='ortho')
    h_osc = h_osc - h_osc.mean()
    h = baseline + h_osc
    return h


def _param_obj(p):
    h = _param_to_h(p)
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e12
    conv = _autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _spectral_init(rng, m_dct):
    """Initialize DCT coefficients using spectral shaping insight.
    
    The autoconvolution c[k] = sum_i h[i]h[i+k] has a peak at lag 0 typically.
    To minimize the peak, we want the spectral density |H(f)|^2 to be as flat 
    as possible (white spectrum), which means h should have flat power spectrum.
    In DCT domain, this means spreading energy across many coefficients.
    
    We use a structured initialization:
    - Low DCT coefficients get moderate amplitude (smooth baseline variation)
    - Some mid coefficients get larger amplitude to create oscillations
    - The pattern is chosen to create a "flat" autoconvolution profile
    """
    coeffs = np.zeros(m_dct)
    
    # Strategy: use a mix of low and mid frequencies with structured amplitudes
    # The key insight from Toeplitz structure: if h has a flat spectrum,
    # the autoconvolution is approximately constant (flat), minimizing the peak.
    
    # Use alternating signs to create a more uniform autoconvolution
    for j in range(m_dct):
        freq = j + 1
        # Amplitude decays slowly to spread energy
        amp = 1.0 / (1.0 + 0.1 * freq)
        # Random sign for diversity
        sign = rng.choice([-1, 1])
        coeffs[j] = sign * amp * rng.uniform(0.3, 1.0)
    
    return coeffs


def construct():
    t0 = time.time()
    DEADLINE = 145.0  # wall-clock budget; keep margin before 150s kill

    def time_left():
        return DEADLINE - (time.time() - t0)

    rng = np.random.default_rng(12345)

    # Number of DCT coefficients to use for the oscillatory part
    m_dct = 30

    # ---- Phase 0: Spectral shaping initialization ----
    # Generate multiple structured initializations using spectral insight
    best_h = np.ones(N, dtype=float)
    best_val = _hard(best_h)

    baselines = [0.5, 1.0, 1.5, -0.5, -1.0, 0.0]
    
    for baseline in baselines:
        if time_left() < 40.0:
            break
        # Multiple random structured inits per baseline
        for trial in range(4):
            if time_left() < 35.0:
                break
            # Mix of spectral init and random
            if trial == 0:
                coeffs = _spectral_init(rng, m_dct)
            else:
                coeffs = rng.normal(0, 0.8, size=m_dct)
                # Add some structure: boost certain frequencies
                for j in [0, 2, 5, 10, 15, 20]:
                    if j < m_dct:
                        coeffs[j] *= 1.5
            
            p0 = np.concatenate([[baseline], coeffs])
            
            try:
                res = minimize(_param_obj, p0, method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10,
                                        'maxfun': 3000})
                h = _param_to_h(res.x)
                v = _hard(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()
            except Exception:
                pass

    # ---- Phase 1: Full 400-dim polish with smooth surrogate (FFT gradient) ----
    h_curr = best_h.copy()
    s = h_curr.sum()
    if abs(s) > 1e-9:
        h_curr = h_curr / s

    beta_seq = [8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
    for beta in beta_seq:
        if time_left() < 20.0:
            break
        try:
            def obj(x):
                val, _ = _obj_and_grad(x, beta)
                return val
            def gr(x):
                _, g = _obj_and_grad(x, beta)
                return g
            res = minimize(obj, h_curr, jac=gr, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12,
                                    'maxfun': 5000})
            h_new = res.x
            v = _hard(h_new)
            if v < best_val:
                best_val = v
                best_h = h_new.copy()
                h_curr = h_new.copy()
        except Exception:
            pass

    # ---- Phase 2: Coordinate / subgradient refinement on hard objective ----
    h_curr = best_h.copy()
    s = h_curr.sum()
    if abs(s) > 1e-9:
        h_curr = h_curr / s
    best_local = _hard(h_curr)
    iters = 0
    while time_left() > 10.0 and iters < 80:
        iters += 1
        conv = _autoconv(h_curr)
        kmax = int(np.argmax(conv))
        n = N
        sg = np.zeros(n)
        lag = kmax - (n - 1)
        for i in range(n):
            a = lag - i
            b = i - lag
            if 0 <= a < n:
                sg[i] += h_curr[a]
            if 0 <= b < n:
                sg[i] += h_curr[b]
        ss = h_curr.sum()
        if abs(ss) < 1e-9:
            break
        sg_centered = sg - sg.mean()
        # Try multiple step sizes with line search
        best_alpha_vv = best_local
        for alpha in [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]:
            cand = h_curr - alpha * sg_centered
            vv = _hard(cand)
            if vv < best_alpha_vv:
                best_alpha_vv = vv
                h_curr = cand
                if vv < best_val:
                    best_val = vv
                    best_h = cand.copy()
        best_local = best_alpha_vv

    # ---- Phase 3: Final high-precision polish with very high beta ----
    if time_left() > 12.0:
        for beta in [256.0, 512.0, 1024.0]:
            if time_left() < 8.0:
                break
            try:
                h_start = best_h.copy()
                ss = h_start.sum()
                if abs(ss) > 1e-9:
                    h_start = h_start / ss
                
                def obj(x):
                    val, _ = _obj_and_grad(x, beta)
                    return val
                def gr(x):
                    _, g = _obj_and_grad(x, beta)
                    return g
                res = minimize(obj, h_start, jac=gr, method='L-BFGS-B',
                               options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-13,
                                        'maxfun': 6000})
                v = _hard(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
            except Exception:
                pass

    # ---- Phase 4: One more coordinate pass if time allows ----
    if time_left() > 6.0:
        h_curr = best_h.copy()
        best_local = _hard(h_curr)
        for _ in range(30):
            if time_left() < 4.0:
                break
            conv = _autoconv(h_curr)
            kmax = int(np.argmax(conv))
            n = N
            sg = np.zeros(n)
            lag = kmax - (n - 1)
            for i in range(n):
                a = lag - i
                b = i - lag
                if 0 <= a < n:
                    sg[i] += h_curr[a]
                if 0 <= b < n:
                    sg[i] += h_curr[b]
            ss = h_curr.sum()
            if abs(ss) < 1e-9:
                break
            sg_centered = sg - sg.mean()
            for alpha in [1e-3, 5e-4, 1e-4, 5e-5]:
                cand = h_curr - alpha * sg_centered
                vv = _hard(cand)
                if vv < best_local:
                    best_local = vv
                    h_curr = cand
                    if vv < best_val:
                        best_val = vv
                        best_h = cand.copy()
                    break

    # ---- Sanity / validity ----
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
        best_val = _hard(best_h)
    if abs(best_h.sum()) < 1e-10:
        best_h = np.ones(N)
        best_val = _hard(best_h)

    return list(best_h)
