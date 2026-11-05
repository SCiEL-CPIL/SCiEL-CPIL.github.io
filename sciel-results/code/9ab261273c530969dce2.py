"""Model-and-optimize program for third autocorrelation inequality.

Strategy:
  * Direct optimization of the 400-dim vector h using FFT-based soft-max
    surrogate of the signed max of the autoconvolution.
  * Multiple diverse initializations (random, structured, DCT-based).
  * Adaptive beta schedule: start low for exploration, increase for exploitation.
  * Final hard-objective subgradient polish.
  * Time-budget aware: returns best result before deadline.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.fftpack import dct, idct

N = 400


def _autoconv(h):
    """Full autoconvolution via FFT: c[k] for lag k = -(N-1)..(N-1), length 2N-1."""
    n = len(h)
    f = np.fft.rfft(h, n=2 * n)
    c = np.fft.irfft(f * f, n=2 * n)
    return c[:2 * n - 1]


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

    # Gradient: grad_i = sum_k w_k * d(conv_k)/d(h_i)
    # d(conv_k)/d(h_i) = h[k-i] (if 0<=k-i<n) + h[i-k] (if 0<=i-k<n)
    # So grad_i = sum_k w_k h[k-i] + sum_k w_k h[i-k]
    
    # Term A: sum_k w_k h[i-k] = (w * h)[i] where * is full convolution, take first n
    # Term B: sum_k w_k h[k-i] = cross-correlation
    
    # Use FFT for efficiency
    W = np.fft.rfft(w, n=2 * n)
    H = np.fft.rfft(h, n=2 * n)
    
    # Term A: convolve(w, h, 'full')[0:n]
    conv_wh = np.fft.irfft(W * H, n=2 * n)
    termA = conv_wh[0:n]
    
    # Term B: sum_k w_k h[k-i] for i=0..n-1
    # This is the cross-correlation of h and w: corr[i] = sum_k h[k] w[k+i]
    # Via FFT: ifft(conj(H) * W) gives sum_k h[k] w[p-k]
    # We need sum_k h[k] w[k+i] = ifft(conj(H)*W) at index (n-1+i) in full correlate
    # Actually: np.correlate(h, w, 'full')[p] = sum_k h[k] w[k+p-(n-1)]
    # So correlate(h,w,'full')[n-1+i] = sum_k h[k] w[k+i] = term B for index i
    
    # For n=400, O(n^2) = 160k ops is fast enough
    B = np.correlate(h, w, mode='full')
    termB = B[n-1:2*n-1]
    
    grad = termA + termB
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


def _dct_init(rng, m_coeffs, baseline=1.0):
    """Create h from DCT coefficients with given baseline."""
    coeffs = np.zeros(N)
    coeffs[0] = baseline
    for j in range(1, min(m_coeffs, N)):
        coeffs[j] = rng.normal(0, 1.0) * (1.0 / (1.0 + 0.05 * j))
    h = idct(coeffs, type=2, norm='ortho')
    return h


def _random_init(rng, scale=1.0):
    """Random initialization with some structure."""
    h = rng.normal(1.0, scale, size=N)
    return h


def _structured_init(rng, pattern='alternating'):
    """Structured initialization based on known patterns."""
    if pattern == 'alternating':
        h = np.ones(N)
        for i in range(N):
            h[i] = 1.0 + 0.5 * np.cos(2 * np.pi * i / 8) + 0.3 * np.cos(2 * np.pi * i / 3)
        return h
    elif pattern == 'sawtooth':
        h = np.ones(N)
        for i in range(N):
            h[i] = 1.0 + 0.4 * np.sin(2 * np.pi * i / 5)
        return h
    elif pattern == 'bump':
        h = np.ones(N)
        for i in range(N):
            x = (i / N) - 0.5
            h[i] = 1.0 + 0.8 * np.exp(-10 * x * x)
        return h
    else:
        return rng.normal(1.0, 0.5, size=N)


def construct():
    t0 = time.time()
    DEADLINE = 145.0

    def time_left():
        return DEADLINE - (time.time() - t0)

    rng = np.random.default_rng(42)

    # Generate diverse initializations
    inits = []
    
    # Random inits with different scales
    for scale in [0.3, 0.5, 0.8, 1.0, 1.5]:
        inits.append(_random_init(rng, scale))
    
    # DCT-based inits
    for m in [10, 20, 30, 50]:
        inits.append(_dct_init(rng, m, baseline=1.0))
    
    # Structured inits
    inits.append(_structured_init(rng, 'alternating'))
    inits.append(_structured_init(rng, 'sawtooth'))
    inits.append(_structured_init(rng, 'bump'))
    
    # Seed
    inits.append(np.ones(N, dtype=float))

    best_h = np.ones(N, dtype=float)
    best_val = _hard(best_h)

    # Evaluate all inits
    for h_init in inits:
        if time_left() < 30.0:
            break
        v = _hard(h_init)
        if v < best_val:
            best_val = v
            best_h = h_init.copy()

    # Main optimization loop with adaptive beta
    h_curr = best_h.copy()
    s = h_curr.sum()
    if abs(s) > 1e-9:
        h_curr = h_curr / s

    # Adaptive beta schedule: start moderate, increase gradually
    beta_schedule = [4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0]
    
    for beta in beta_schedule:
        if time_left() < 15.0:
            break
        
        # Try optimizing from current best and from a few other inits
        starts = [h_curr.copy()]
        # Add 2 random restarts for diversity
        for _ in range(2):
            h_rand = _random_init(rng, 0.5)
            sr = h_rand.sum()
            if abs(sr) > 1e-9:
                h_rand = h_rand / sr
            starts.append(h_rand)
        
        for h_start in starts:
            if time_left() < 10.0:
                break
            try:
                def obj(x):
                    val, _ = _obj_and_grad(x, beta)
                    return val
                def gr(x):
                    _, g = _obj_and_grad(x, beta)
                    return g
                
                res = minimize(obj, h_start, jac=gr, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12,
                                        'maxfun': 8000})
                h_new = res.x
                v = _hard(h_new)
                if v < best_val:
                    best_val = v
                    best_h = h_new.copy()
                    h_curr = h_new.copy()
            except Exception:
                pass

    # Subgradient polish on hard objective
    h_curr = best_h.copy()
    s = h_curr.sum()
    if abs(s) > 1e-9:
        h_curr = h_curr / s
    best_local = _hard(h_curr)
    
    iters = 0
    while time_left() > 8.0 and iters < 100:
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
        
        # Try multiple step sizes
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

    # Final high-beta polish if time allows
    if time_left() > 10.0:
        for beta in [512.0, 1024.0]:
            if time_left() < 6.0:
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
                               options={'maxiter': 600, 'ftol': 1e-15, 'gtol': 1e-13,
                                        'maxfun': 10000})
                v = _hard(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
            except Exception:
                pass

    # Sanity checks
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
        best_val = _hard(best_h)
    if abs(best_h.sum()) < 1e-10:
        best_h = np.ones(N)
        best_val = _hard(best_h)

    return list(best_h)
