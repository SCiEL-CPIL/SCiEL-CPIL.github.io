"""Erdos minimum overlap: vectorized smooth-surrogate optimization.
Uses log-sum-exp soft-max of the overlap values as a differentiable surrogate
for the max, optimized with L-BFGS-B using penalty for sum constraint,
with vectorized gradient via FFT and multiple diverse starts.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 145  # Safety margin before 150s kill
_TARGET = N / 2.0
_NFFT = 256  # next power of 2 >= 2*N


def _autocorr(h):
    """Linear autocorrelation r[k] = sum_{i=k}^{N-1} h[i]*h[i-k] for k=0..N-1 via FFT."""
    h_pad = np.zeros(_NFFT)
    h_pad[:N] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H))
    return corr[:N]


def _overlap_values(h):
    """C_k = (2/N)*(N/2 - r_k) for k=0..N-1."""
    n = len(h)
    r = _autocorr(h)
    return (2.0 / n) * (n / 2.0 - r)


def _score(h):
    """True (hard) maximum overlap value."""
    cs = _overlap_values(h)
    return float(np.max(cs))


def _project(h):
    """Project h to sum=N/2, values in [0,1] (Euclidean-ish projection)."""
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
    # Final exact correction
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
    grad = e / s  # gradient w.r.t. x
    return val, grad


def _smooth_obj_and_grad(h, beta):
    """f(h) = S_beta(C(h)); returns (f, df/dh). Vectorized via FFT."""
    cs = _overlap_values(h)
    val, g_c = _smooth_max_and_grad(cs, beta)
    # Gradient: g_h[j] = factor * sum_k g_c[k] * d r_k / d h_j
    # d r_k / d h_j = h[j-k] (j>=k) + h[j+k] (j+k<N) for k>0, and 2*h[j] for k=0
    # This is equivalent to: g_h = factor * (2*g_c[0]*h + 2*convolve(h, g_c[1:]))
    # where convolve is the linear cross-correlation: (h * g_c)[j] = sum_k h[j-k]*g_c[k]
    # Actually: sum_k g_c[k] * (h[j-k] + h[j+k]) for k>=1
    # = sum_k g_c[k]*h[j-k] + sum_k g_c[k]*h[j+k]
    # The first term is cross-correlation of h with g_c[1:]
    # The second term: sum_k g_c[k]*h[j+k] = sum_k g_c[k]*h[(j+k)] 
    # Let m = j+k, then k = m-j, sum_m g_c[m-j]*h[m] for m-j>=1
    # This is also a cross-correlation.
    # Actually, sum_k g_c[k]*h[j+k] for k=1..N-1 is the same as sum_k g_c[N-1-k]*h[j-(N-1-k)]
    # Let's use: g_h[j] = factor * (2*g_c[0]*h[j] + 2 * sum_{k=1}^{N-1} g_c[k]*h[j-k])
    # Wait, need to be careful. Let me re-derive.
    # r_k = sum_{i=k}^{N-1} h[i]*h[i-k]
    # d r_k / d h_j: h[j] appears in r_k when j>=k (as h[j]*h[j-k]) and when j+k<N (as h[j+k]*h[j])
    # So d r_k/d h_j = h[j-k] (if j>=k) + h[j+k] (if j+k<N)
    # For k=0: d r_0/d h_j = 2*h[j]
    # g_h[j] = factor * sum_{k=0}^{N-1} g_c[k] * d r_k/d h_j
    # = factor * (2*g_c[0]*h[j] + sum_{k=1}^{N-1} g_c[k]*(h[j-k] + h[j+k]))
    # The sum over k>=1 of g_c[k]*h[j-k] is a cross-correlation.
    # Let g = g_c[1:], then sum_{k=1}^{N-1} g_c[k]*h[j-k] = sum_{m=0}^{N-2} g[m]*h[j-1-m]
    # This is the convolution of h with reversed g, evaluated at j-1.
    # Let's compute via FFT for efficiency.
    n = N
    # Compute cross-correlation: cc[j] = sum_{k=1}^{N-1} g_c[k] * h[j-k] (with h out of range = 0)
    # This is the linear cross-correlation of h and g_c[1:]
    g = g_c[1:]
    # cc[j] = sum_k g[k]*h[j-k-1] for k=0..N-2, which is conv(h, g)[j-1]
    # Use FFT: pad both to _NFFT
    h_pad = np.zeros(_NFFT)
    h_pad[:N] = h
    g_pad = np.zeros(_NFFT)
    g_pad[:N-1] = g
    H = np.fft.rfft(h_pad)
    G = np.fft.rfft(g_pad)
    # Cross-correlation: corr[j] = sum_k h[j+k]*g[k] = ifft(conj(H)*G)
    # We want sum_k g[k]*h[j-k] which is the same as corr with h and g swapped
    # corr_hg[j] = sum_k h[j-k]*g[k] = ifft(conj(H)*G)
    corr_hg = np.fft.irfft(np.conj(H) * G)
    # corr_hg[j] = sum_k h[j-k]*g[k] for appropriate range
    # We need sum_{k=1}^{N-1} g_c[k]*h[j-k] = sum_{m=0}^{N-2} g[m]*h[j-1-m]
    # = corr_hg[j-1] where corr_hg is computed with g = g_c[1:]
    # Actually corr_hg[j] = sum_k h[j-k]*g[k], so corr_hg[j-1] = sum_k h[j-1-k]*g[k] = sum_k g[k]*h[j-1-k]
    # That's what we want for the first part.
    part1 = corr_hg[:N]  # part1[j] = sum_k g[k]*h[j-k] for j=0..N-1
    # But we need sum_k g[k]*h[j-k] where g[k]=g_c[k+1], so this is sum_{k=0}^{N-2} g_c[k+1]*h[j-k]
    # = sum_{m=1}^{N-1} g_c[m]*h[j-m+1]... hmm let me re-index
    # Let me just do: part1[j] should be sum_{k=1}^{N-1} g_c[k]*h[j-k]
    # corr_hg[j] = sum_{k=0}^{N-2} h[j-k]*g[k] = sum_{k=0}^{N-2} g_c[k+1]*h[j-k]
    # = sum_{m=1}^{N-1} g_c[m]*h[j-m+1]... no
    # corr_hg[j] = sum_{k=0}^{N-2} h[j-k] * g_c[k+1]
    # We want sum_{k=1}^{N-1} g_c[k] * h[j-k] = sum_{m=0}^{N-2} g_c[m+1] * h[j-m-1]
    # = corr_hg[j-1]
    # So part1[j] = corr_hg[j-1] for j>=1, and part1[0] = 0 (no valid k)
    part1 = np.zeros(N)
    part1[1:] = corr_hg[:N-1]
    
    # Second part: sum_{k=1}^{N-1} g_c[k]*h[j+k]
    # = sum_{k=1}^{N-1} g_c[k]*h[j+k]
    # Let l = j+k, then k = l-j, sum_{l=j+1}^{j+N-1} g_c[l-j]*h[l]
    # = sum_{m=1}^{N-1} g_c[m]*h[j+m]
    # This is like a cross-correlation with shifted indices.
    # sum_m g_c[m]*h[j+m] = sum_m g_c[N-1-m]*h[j-(N-1-m)]... 
    # Let's compute: part2[j] = sum_{m=1}^{N-1} g_c[m]*h[j+m]
    # = sum_{m=0}^{N-2} g_c[m+1]*h[j+m+1]
    # Let n = m+1: sum_{n=1}^{N-1} g_c[n]*h[j+n]
    # This is the same as part1 but with h shifted. 
    # part2[j] = sum_{n=1}^{N-1} g_c[n]*h[j+n] = sum_{n=1}^{N-1} g_c[n]*h[(j+n)]
    # Note that part1[N-1-j] = sum_{n=1}^{N-1} g_c[n]*h[N-1-j-n]
    # Hmm, let me just compute it directly via FFT.
    # part2[j] = sum_{n=1}^{N-1} g_c[n]*h[j+n]
    # Let's define g_rev where g_rev[N-1-n] = g_c[n] for n=1..N-1
    # Then part2[j] = sum_{n=1}^{N-1} g_c[n]*h[j+n] = sum_{n=0}^{N-1} g_rev[n]*h[j+n-(N-1)]... 
    # This is getting complicated. Let me just use the fact that:
    # sum_{n=1}^{N-1} g_c[n]*h[j+n] = sum_{n=1}^{N-1} h[j+n]*g_c[n]
    # This is the cross-correlation of h and g_c at offset -(j+1)... 
    # Actually, let's just compute part2 using the same FFT trick.
    # part2[j] = sum_{n=1}^{N-1} g_c[n]*h[j+n]
    # = sum_{n=0}^{N-2} g_c[n+1]*h[j+n+1]
    # Let's define h_shifted[j] = h[j+1] (with h_shifted[N-1]=0)
    # Then part2[j] = sum_{n=0}^{N-2} g[n]*h_shifted[j+n] where g=g_c[1:]
    # This is a convolution: (g * h_shifted)[j] but in the correlation sense.
    # Actually it's the same as corr_hg but with h shifted by 1.
    h_shift_pad = np.zeros(_NFFT)
    h_shift_pad[:N-1] = h[1:]
    HS = np.fft.rfft(h_shift_pad)
    corr_hs = np.fft.irfft(np.conj(HS) * G)
    # corr_hs[j] = sum_k h_shifted[j-k]*g[k] = sum_k h[j+1-k]*g[k]
    # We want sum_k g[k]*h[j+1+k]... no
    # part2[j] = sum_{n=0}^{N-2} g[n]*h[j+n+1]
    # corr_hs[j] = sum_{k=0}^{N-2} h_shifted[j-k]*g[k] = sum_{k=0}^{N-2} h[j+1-k]*g[k]
    # That's not quite right. Let me try a different approach.
    # Just compute part2 directly using the conjugate:
    # part2[j] = sum_{n=1}^{N-1} g_c[n]*h[j+n]
    # = sum_{n=1}^{N-1} h[j+n]*g_c[n]
    # Let's use: ifft(H * conj(G)) gives sum_k h[k]*g[N-1-k+j]... 
    # I'll just compute it numerically since N=100 is small.
    part2 = np.zeros(N)
    for j in range(N):
        idx = j + 1 + np.arange(N-1)
        valid = idx < N
        part2[j] = np.sum(g_c[1:][valid] * h[idx[valid]])
    
    g_h = -2.0 / N * (2.0 * g_c[0] * h + part1 + part2)
    return val, g_h


def _epigraph_obj_and_grad(x, beta):
    """x = [h (N), t (1)]. Objective = t + S_beta(C(h) - t).
    Returns (obj, grad)."""
    h = x[:N]
    t = x[N]
    cs = _overlap_values(h)
    shifted = cs - t
    sm, g_c = _smooth_max_and_grad(shifted, beta)
    obj = t + sm
    g_h = np.zeros(N)
    factor = -(2.0 / N)
    for k in range(N):
        gc = g_c[k]
        if gc == 0.0:
            continue
        if k == 0:
            g_h += factor * gc * (2.0 * h)
        else:
            g_h[k:] += factor * gc * h[:N - k]
            g_h[:N - k] += factor * gc * h[k:]
    g_t = 1.0 - np.sum(g_c)
    grad = np.concatenate([g_h, [g_t]])
    return obj, grad


def _refine_smooth(h0, beta, n_max_iter=5000, time_budget=30.0):
    """Refine h using L-BFGS-B on the smooth surrogate with box constraints.
    Uses penalty for sum constraint to allow L-BFGS-B (faster than SLSQP).
    """
    h0 = _project(h0.copy())
    h0 = h0.copy()

    penalty = 100.0

    def _obj_grad(h):
        h_clipped = np.clip(h, 0.0, 1.0)
        val, g_h = _smooth_obj_and_grad(h_clipped, beta)
        # Penalty for sum constraint
        s = np.sum(h_clipped)
        pen = penalty * (s - _TARGET) ** 2
        g_pen = 2.0 * penalty * (s - _TARGET)
        return val + pen, g_h + g_pen

    bounds = [(0.0, 1.0)] * N
    t_start = time.time()
    res = minimize(
        _obj_grad, h0, jac=True, method='L-BFGS-B', bounds=bounds,
        options={'maxiter': n_max_iter, 'ftol': 1e-14, 'gtol': 1e-10,
                 'maxls': 50, 'maxfun': 50000},
    )
    if time.time() - t_start > time_budget:
        return h0, _score(h0)
    h_opt = _project(res.x)
    return h_opt, _score(h_opt)


def _construct_direct():
    """Direct multi-frequency construction to flatten autocorrelation."""
    i = np.arange(N)
    h = np.full(N, 0.5)
    # Use frequencies that are well-distributed to minimize peak autocorrelation
    freqs = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]
    amplitudes = [0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
    phases = [0.0, 0.7, 1.3, 2.1, 0.4, 1.8, 2.5, 0.9, 1.6, 0.3]
    for f, a, p in zip(freqs, amplitudes, phases):
        h += a * np.cos(2.0 * np.pi * f * i / N + p)
    return _project(h)


def _random_start(rng):
    """Random feasible start."""
    h = rng.uniform(0.0, 1.0, N)
    return _project(h)


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
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

    # Phase 1: Direct construction
    h_direct = _construct_direct()
    _update(h_direct)

    # Phase 2: Several diverse frequency-based starts
    if time.time() - t0 < _DEADLINE - 50:
        i = np.arange(N)
        freq_sets = [
            ([1, 2, 3, 5], [0.15, 0.10, 0.08, 0.06]),
            ([1, 2, 4, 7], [0.14, 0.10, 0.07, 0.05]),
            ([1, 3, 4, 6, 9], [0.12, 0.10, 0.08, 0.06, 0.04]),
            ([2, 3, 5, 8, 13], [0.12, 0.10, 0.08, 0.06, 0.04]),
            ([1, 2, 3, 4, 5, 6], [0.10, 0.08, 0.06, 0.05, 0.04, 0.03]),
            ([3, 5, 7, 11], [0.12, 0.10, 0.08, 0.06]),
            ([1, 4, 6, 9, 12], [0.11, 0.09, 0.07, 0.05, 0.03]),
            ([2, 5, 8, 13, 21], [0.10, 0.08, 0.06, 0.04, 0.02]),
        ]
        for freqs, amps in freq_sets:
            if time.time() - t0 > _DEADLINE - 50:
                break
            h = np.full(N, 0.5)
            for f, a in zip(freqs, amps):
                h += a * np.cos(2.0 * np.pi * f * i / N)
            _update(h)

    # Phase 3: Random starts
    if time.time() - t0 < _DEADLINE - 40:
        for _ in range(6):
            if time.time() - t0 > _DEADLINE - 40:
                break
            h_rand = _random_start(rng)
            _update(h_rand)

    # Phase 4: Smooth surrogate refinement with L-BFGS-B (fast)
    beta_list = [10.0, 30.0, 100.0, 300.0]
    starts_to_refine = []
    if best_h is not None:
        starts_to_refine.append(best_h.copy())

    # Add more diverse starts
    if time.time() - t0 < _DEADLINE - 80:
        for _ in range(3):
            if time.time() - t0 > _DEADLINE - 80:
                break
            starts_to_refine.append(_random_start(rng))

    for beta in beta_list:
        remaining = _DEADLINE - time.time()
        if remaining < 10:
            break
        # Allocate time per start
        n_starts = len(starts_to_refine)
        time_per_start = min(12.0, (remaining - 5.0) / max(n_starts, 1))
        if time_per_start < 3.0:
            break
        for h_start in starts_to_refine:
            if time.time() - t0 > _DEADLINE - 10:
                break
            h_ref, s_ref = _refine_smooth(h_start, beta=beta, n_max_iter=5000,
                                          time_budget=time_per_start)
            _update(h_ref)
        # Update starts with best for next beta
        if best_h is not None:
            starts_to_refine[0] = best_h.copy()

    # Phase 5: Final high-precision polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 8:
        h_polish, s_polish = _refine_smooth(best_h, beta=500.0, n_max_iter=10000,
                                            time_budget=min(15.0, _DEADLINE - time.time() - 3.0))
        _update(h_polish)

    if best_h is None:
        best_h = np.full(N, 0.5)

    return _project(best_h)
