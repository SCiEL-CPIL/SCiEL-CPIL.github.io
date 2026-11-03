"""Erdos minimum overlap: vectorized smooth-surrogate + bisection on t.
Uses log-sum-exp soft-max of (C(h) - t) as a differentiable surrogate,
with bisection on t and projected gradient / L-BFGS-B for h.
Gradient of the smooth max is computed via FFT-convolution for speed.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_TARGET = N / 2.0
_NFFT = 256  # next power of 2 >= 2*N

# Precompute FFT of basis for gradient
# d r_k / d h_j: for each k, the "kernel" is:
#   for j in [k, N-1]: h[j-k]  -> shift of h
#   for j in [0, N-k-1]: h[j+k] -> shift of h
# We can compute g_h = sum_k g_c[k] * (shifted contributions)
# More efficiently: note that the full gradient is a convolution-like operation.
# Let's think: g_h[j] = factor * sum_k g_c[k] * (h[j-k] * 1_{j>=k} + h[j+k] * 1_{j+k<N})
# This is: factor * sum_k g_c[k] * h[j-k] * 1_{j>=k} + factor * sum_k g_c[k] * h[j+k] * 1_{j+k<N}
# Let a[k] = g_c[k] for k=0..N-1
# First term: sum_{k=0}^{j} a[k] * h[j-k]  = (a * h)[j] (linear convolution, but only up to j)
# Second term: sum_{k=0}^{N-1-j} a[k] * h[j+k]
# Actually let's just use the direct approach but vectorized with FFT.

def _autocorr(h):
    """Linear autocorrelation r[k] = sum_{i=k}^{N-1} h[i]*h[i-k] for k=0..N-1 via FFT."""
    h_pad = np.zeros(_NFFT)
    h_pad[:N] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H), n=_NFFT)
    return corr[:N]

def _overlap_values(h):
    """C_k = (2/N)*(N/2 - r_k) for k=0..N-1."""
    r = _autocorr(h)
    return (2.0 / N) * (N / 2.0 - r)

def _score(h):
    """True (hard) maximum overlap value."""
    cs = _overlap_values(h)
    return float(np.max(cs))

def _project(h):
    """Project h to sum=N/2, values in [0,1]."""
    h = np.array(h, dtype=np.float64, copy=True)
    h = np.clip(h, 0.0, 1.0)
    diff = _TARGET - np.sum(h)
    if abs(diff) > 1e-14:
        if diff > 0:
            mask = h < 1.0
        else:
            mask = h > 0.0
        cnt = np.sum(mask)
        if cnt > 0:
            h[mask] += diff / cnt
            h = np.clip(h, 0.0, 1.0)
    diff = _TARGET - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h

def _smooth_max_and_grad(x, beta):
    """Log-sum-exp soft-max of x and its gradient (softmax(beta*x))."""
    m = np.max(x)
    e = np.exp(beta * (x - m))
    s = np.sum(e)
    val = (m + np.log(s)) / beta
    grad = e / s
    return val, grad

def _grad_smooth_max_wrt_h(h, g_c, beta):
    """Compute gradient of S_beta(C(h)) w.r.t. h, given g_c = dS/dC.
    Uses the structure: dC_k/dh_j = -(2/N) * (h[j-k] if j>=k else 0 + h[j+k] if j+k<N else 0)
    Vectorized via two convolutions.
    """
    factor = -(2.0 / N)
    # Term 1: sum_{k=0}^{j} g_c[k] * h[j-k] for each j  (this is linear cross-correlation)
    # Term 2: sum_{k=0}^{N-1-j} g_c[k] * h[j+k] for each j
    # We can compute both via FFT.
    
    # For term 1: it's the cross-correlation of g_c and h, but only the "causal" part
    # Actually, let's think differently. Define:
    # term1[j] = sum_{k=0}^{min(j, N-1)} g_c[k] * h[j-k]
    # This is the linear convolution (g_c * h)[j] but truncated.
    # Since both have length N, the linear convolution has length 2N-1.
    # term1[j] = conv[j] for j=0..N-1
    
    # For term 2: term2[j] = sum_{k=0}^{N-1-j} g_c[k] * h[j+k]
    # Let's reindex: let m = N-1-j, then term2[j] = sum_{k=0}^{m} g_c[k] * h[N-1-m+k]
    # This is like a "reverse convolution". 
    # Alternatively: term2[j] = sum_{k=0}^{N-1} g_c[k] * h[j+k] * 1_{j+k < N}
    # = sum_{k=0}^{N-1-j} g_c[k] * h[j+k]
    # This is the same as: if we reverse h, then it's a cross-correlation.
    # Let hr = h[::-1], then term2[j] = sum_{k=0}^{N-1-j} g_c[k] * hr[N-1-j-k]
    # = (g_c * hr)[N-1-j]
    
    # Compute via FFT:
    gc_pad = np.zeros(_NFFT)
    gc_pad[:N] = g_c
    h_pad = np.zeros(_NFFT)
    h_pad[:N] = h
    hr_pad = np.zeros(_NFFT)
    hr_pad[:N] = h[::-1]
    
    GC = np.fft.rfft(gc_pad)
    HC = np.fft.rfft(h_pad)
    HRC = np.fft.rfft(hr_pad)
    
    conv1 = np.fft.irfft(GC * HC, n=_NFFT)[:N]  # term1
    conv2_full = np.fft.irfft(GC * HRC, n=_NFFT)  # full convolution for term2
    # term2[j] = conv2_full[N-1-j]
    term2 = conv2_full[N-1::-1][:N]
    
    g_h = factor * (conv1 + term2)
    return g_h

def _construct_direct():
    """Direct multi-frequency construction to flatten autocorrelation."""
    i = np.arange(N)
    h = np.zeros(N)
    h += 0.5
    freqs = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0]
    amplitudes = [0.15, 0.10, 0.08, 0.06, 0.04, 0.03]
    phases = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    for f, a, p in zip(freqs, amplitudes, phases):
        h += a * np.cos(2.0 * np.pi * f * i / N + p)
    return h

def _random_start(rng):
    """Random feasible start."""
    h = rng.uniform(0.0, 1.0, N)
    return _project(h)

def _projected_gradient_step(h, g_h, step_size):
    """One projected gradient step with step_size."""
    h_new = h - step_size * g_h
    return _project(h_new)

def _nesterov_projected_gradient(h0, beta, n_iter=200, step_size=0.01):
    """Nesterov accelerated projected gradient on smooth-max surrogate."""
    h = h0.copy()
    h_prev = h0.copy()
    gamma = 0.5
    for it in range(n_iter):
        # Nesterov lookahead
        h_next = h + gamma * (h - h_prev)
        cs = _overlap_values(h_next)
        _, g_c = _smooth_max_and_grad(cs, beta)
        g_h = _grad_smooth_max_wrt_h(h_next, g_c, beta)
        h_new = _projected_gradient_step(h_next, g_h, step_size)
        gamma = (1 + np.sqrt(1 + 4 * gamma * gamma)) / 2
        h_prev = h
        h = h_new
    return h

def _refine_lbfgsb(h0, beta, time_budget=20.0):
    """Refine using L-BFGS-B with penalty for sum constraint."""
    h0 = _project(h0.copy())
    
    def _obj_grad(h):
        cs = _overlap_values(h)
        val, g_c = _smooth_max_and_grad(cs, beta)
        g_h = _grad_smooth_max_wrt_h(h, g_c, beta)
        # Add penalty for sum constraint
        sum_penalty = 100.0 * (np.sum(h) - _TARGET)
        val += sum_penalty
        g_h = g_h + 100.0 * np.ones(N)
        return val, g_h
    
    bounds = [(0.0, 1.0)] * N
    t_start = time.time()
    res = minimize(
        _obj_grad, h0, jac=True, method='L-BFGS-B', bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-12, 'disp': False, 'maxls': 30}
    )
    if time.time() - t_start > time_budget:
        return h0, _score(h0)
    h_opt = _project(res.x)
    return h_opt, _score(h_opt)

def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    deadline = t0 + 140  # Safety margin before 150s kill
    rng = np.random.default_rng(42)

    best_h = None
    best_score = float('inf')

    def _update(h):
        nonlocal best_h, best_score
        h = _project(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
        return s

    # Phase 1: Direct construction
    h_direct = _construct_direct()
    _update(h_direct)

    # Phase 2: Several diverse frequency-based starts
    if time.time() < deadline - 60:
        i = np.arange(N)
        freq_sets = [
            ([1, 2, 3, 5], [0.15, 0.10, 0.08, 0.06]),
            ([1, 2, 4, 7], [0.14, 0.10, 0.07, 0.05]),
            ([1, 3, 4, 6, 9], [0.12, 0.10, 0.08, 0.06, 0.04]),
            ([2, 3, 5, 8, 13], [0.12, 0.10, 0.08, 0.06, 0.04]),
            ([1, 2, 3, 4, 5, 6], [0.10, 0.08, 0.06, 0.05, 0.04, 0.03]),
            ([3, 5, 7, 11], [0.12, 0.10, 0.08, 0.06]),
            ([1, 2, 3, 4, 5], [0.12, 0.10, 0.08, 0.06, 0.05]),
            ([2, 4, 6, 8, 10], [0.10, 0.09, 0.08, 0.07, 0.06]),
        ]
        for freqs, amps in freq_sets:
            if time.time() > deadline - 60:
                break
            h = np.full(N, 0.5)
            for f, a in zip(freqs, amps):
                h += a * np.cos(2.0 * np.pi * f * i / N)
            _update(h)

    # Phase 3: Random starts
    if time.time() < deadline - 50:
        for _ in range(6):
            if time.time() > deadline - 50:
                break
            h_rand = _random_start(rng)
            _update(h_rand)

    # Phase 4: Nesterov projected gradient refinement from best starts
    beta_list = [10.0, 20.0, 50.0, 100.0]
    starts_to_refine = []
    if best_h is not None:
        starts_to_refine.append(best_h.copy())
    
    # Add a couple more diverse starts
    if time.time() < deadline - 80:
        for _ in range(3):
            if time.time() > deadline - 80:
                break
            starts_to_refine.append(_random_start(rng))
    
    for h_start in starts_to_refine:
        for beta in beta_list:
            if time.time() > deadline - 30:
                break
            h_nest = _nesterov_projected_gradient(h_start, beta, n_iter=150, step_size=0.005)
            _update(h_nest)

    # Phase 5: L-BFGS-B refinement from best
    if best_h is not None and time.time() < deadline - 20:
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 10:
                break
            h_ref, s_ref = _refine_lbfgsb(best_h, beta, time_budget=15.0)
            _update(h_ref)

    # Phase 6: Final polish with very high beta
    if best_h is not None and time.time() < deadline - 5:
        h_polish = _nesterov_projected_gradient(best_h, 500.0, n_iter=100, step_size=0.002)
        _update(h_polish)
        h_ref, s_ref = _refine_lbfgsb(best_h, 500.0, time_budget=10.0)
        _update(h_ref)

    if best_h is None:
        best_h = np.full(N, 0.5)

    return _project(best_h)
