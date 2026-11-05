"""Model-and-optimize program for third autocorrelation inequality.

Strategy:
  * Represent h as a uniform background plus a small number of high-amplitude
    spikes at chosen positions.  This gives a low-dimensional parameterization
    (spike positions + amplitudes + background) that directly targets the
    ratio of peak autoconvolution to (sum)^2.
  * Use multi-start random search over spike configurations, followed by
    L-BFGS-B refinement on the full 400-dim vector from the best sparse
    initialization, then coordinate/subgradient polish on the hard objective.
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


def _hard(h):
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e12
    conv = _autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _spike_to_h(a, positions, amplitudes):
    """Build h from background a, spike positions (int array), and amplitudes."""
    h = np.full(N, a, dtype=float)
    for pos, amp in zip(positions, amplitudes):
        h[pos] += amp
    return h


def _spike_obj(params):
    """Objective for sparse spike parameterization.
    params = [a, pos_1, pos_2, ..., pos_m, amp_1, amp_2, ..., amp_m]
    """
    m = len(params) // 2
    a = params[0]
    positions = params[1:1+m].astype(int)
    amplitudes = params[1+m:1+2*m]
    # Ensure positions are valid and unique
    h = np.full(N, a, dtype=float)
    for i in range(m):
        pos = int(positions[i]) % N
        h[pos] += amplitudes[i]
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e12
    conv = _autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))


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


def construct():
    t0 = time.time()
    DEADLINE = 145.0

    def time_left():
        return DEADLINE - (time.time() - t0)

    rng = np.random.default_rng(42)

    best_h = np.ones(N, dtype=float)
    best_val = _hard(best_h)  # 2.0

    # ---- Phase 0: Sparse spike search ----
    # h = a (background) + spikes at chosen positions with chosen amplitudes
    # Parameterize: m spikes, each with position (0..N-1) and amplitude
    # Key idea: if all h[i] = a, then c[k] = (N-|k|)*a^2, max at k=0: c[0]=N*a^2
    #   value = 2*N*N*a^2 / (N*a)^2 = 2.0
    # If we add spikes, we want to reduce the peak relative to sum^2.
    # Strategy: use negative spikes to reduce c[0] while keeping sum positive,
    # or use a mix of positive and negative spikes to flatten the autoconvolution.

    m_spikes_list = [2, 3, 4, 5, 6, 8]
    
    for m in m_spikes_list:
        if time_left() < 30.0:
            break
        n_trials = max(20, int(60 * time_left() / 120.0))
        
        for trial in range(n_trials):
            if time_left() < 25.0:
                break
            # Random spike configuration
            a_bg = rng.uniform(0.1, 2.0)
            positions = rng.choice(N, size=m, replace=False)
            # Amplitudes: mix of positive and negative, larger magnitude than bg
            amps = rng.uniform(-5.0, 5.0, size=m)
            # Ensure sum is not near zero
            total = a_bg * N + amps.sum()
            if abs(total) < 0.1:
                continue
            
            h = np.full(N, a_bg, dtype=float)
            for pos, amp in zip(positions, amps):
                h[pos] += amp
            
            v = _hard(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()

            # Try a few L-BFGS-B refinements from this sparse init
            if v < best_val * 0.95 and time_left() > 20.0:
                try:
                    s = h.sum()
                    if abs(s) > 1e-9:
                        h_norm = h / s
                    else:
                        h_norm = h.copy()
                    
                    # Quick L-BFGS-B on full vector with moderate beta
                    def obj_quick(x):
                        val, _ = _obj_and_grad(x, 16.0)
                        return val
                    def gr_quick(x):
                        _, g = _obj_and_grad(x, 16.0)
                        return g
                    res = minimize(obj_quick, h_norm, jac=gr_quick, method='L-BFGS-B',
                                   options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8,
                                            'maxfun': 1500})
                    v2 = _hard(res.x)
                    if v2 < best_val:
                        best_val = v2
                        best_h = res.x.copy()
                except Exception:
                    pass

    # ---- Phase 1: Structured spike patterns ----
    # Try symmetric and evenly-spaced spike patterns
    for m in [2, 3, 4, 5, 6, 8, 10, 12]:
        if time_left() < 25.0:
            break
        for a_bg in [0.5, 1.0, 1.5]:
            if time_left() < 20.0:
                break
            # Evenly spaced
            positions = np.linspace(0, N-1, m, dtype=int)
            for amp_scale in [1.0, 2.0, 3.0, 5.0]:
                for sign_pattern in ['+--+', '-++-', '+-+-', '+-+-']:
                    amps = np.array([
                        amp_scale * (1 if sign_pattern[i % len(sign_pattern)] == '+' else -1)
                        for i in range(m)
                    ])
                    total = a_bg * N + amps.sum()
                    if abs(total) < 0.1:
                        continue
                    h = np.full(N, a_bg, dtype=float)
                    for pos, amp in zip(positions, amps):
                        h[pos] += amp
                    v = _hard(h)
                    if v < best_val:
                        best_val = v
                        best_h = h.copy()
                    
                    # Refine
                    if v < best_val * 0.95 and time_left() > 15.0:
                        try:
                            s = h.sum()
                            if abs(s) > 1e-9:
                                h_norm = h / s
                            else:
                                h_norm = h.copy()
                            def obj_q(x):
                                val, _ = _obj_and_grad(x, 32.0)
                                return val
                            def gr_q(x):
                                _, g = _obj_and_grad(x, 32.0)
                                return g
                            res = minimize(obj_q, h_norm, jac=gr_q, method='L-BFGS-B',
                                           options={'maxiter': 200, 'ftol': 1e-11, 'gtol': 1e-9,
                                                    'maxfun': 3000})
                            v2 = _hard(res.x)
                            if v2 < best_val:
                                best_val = v2
                                best_h = res.x.copy()
                        except Exception:
                            pass

    # ---- Phase 2: Full 400-dim polish with smooth surrogate ----
    h_curr = best_h.copy()
    s = h_curr.sum()
    if abs(s) > 1e-9:
        h_curr = h_curr / s

    beta_seq = [8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
    for beta in beta_seq:
        if time_left() < 15.0:
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

    # ---- Phase 3: Coordinate / subgradient refinement on hard objective ----
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

    # ---- Phase 4: Final high-precision polish ----
    if time_left() > 10.0:
        for beta in [256.0, 512.0, 1024.0]:
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
                               options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-13,
                                        'maxfun': 6000})
                v = _hard(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
            except Exception:
                pass

    # ---- Phase 5: Final coordinate pass ----
    if time_left() > 5.0:
        h_curr = best_h.copy()
        best_local = _hard(h_curr)
        for _ in range(30):
            if time_left() < 3.0:
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
