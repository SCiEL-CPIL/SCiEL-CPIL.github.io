"""Erdos minimum overlap: smooth-surrogate optimization with vectorized gradients.
Uses log-sum-exp soft-max surrogate, Nesterov accelerated projected gradient,
and a final high-precision polish. Key improvement: vectorized gradient computation
via FFT-based convolution to eliminate the O(N^2) Python loop.
"""
import time
import numpy as np

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
    grad = e / s
    return val, grad


def _smooth_obj_and_grad(h, beta):
    """f(h) = S_beta(C(h)); returns (f, df/dh). Vectorized via FFT."""
    cs = _overlap_values(h)
    val, g_c = _smooth_max_and_grad(cs, beta)
    
    # C_k = (2/N)*(N/2 - r_k) where r_k = sum_i h[i+k]*h[i] (for k>=0, with h[i+k]=0 for i+k>=N)
    # dC_k/dh[j] = -(2/N) * (h[j-k] + h[j+k] - boundary corrections)
    # More precisely: r_k = sum_{i=0}^{N-1-k} h[i+k]*h[i]
    # dr_k/dh[j] = h[j-k] (if j>=k) + h[j+k] (if j+k<N)
    # So dC_k/dh[j] = -(2/N) * (h[j-k]*[j>=k] + h[j+k]*[j+k<N])
    
    # Vectorized: for each k, the contribution to g_h is g_c[k] * dC_k/dh
    # g_h[j] = sum_k g_c[k] * dC_k/dh[j]
    #        = -(2/N) * sum_k g_c[k] * (h[j-k]*[j>=k] + h[j+k]*[j+k<N])
    
    # This is a convolution-like operation. Let's compute it efficiently.
    # term1[j] = sum_{k=0}^{j} g_c[k] * h[j-k]  (this is (g_c * h)[j] truncated)
    # term2[j] = sum_{k=0}^{N-1-j} g_c[k] * h[j+k]  (this is like a cross-correlation)
    
    # Use FFT for the convolution
    g_c_pad = np.zeros(_NFFT)
    g_c_pad[:N] = g_c
    h_pad = np.zeros(_NFFT)
    h_pad[:N] = h
    
    G = np.fft.rfft(g_c_pad)
    H = np.fft.rfft(h_pad)
    
    # term1: (g_c * h)[j] for j=0..N-1
    conv = np.fft.irfft(G * H)
    term1 = conv[:N]  # sum_{k=0}^{j} g_c[k]*h[j-k]
    
    # term2[j] = sum_{k=0}^{N-1-j} g_c[k]*h[j+k]
    # This is like a flipped cross-correlation. Let's compute it differently.
    # term2[j] = sum_{m=j}^{N-1} g_c[m-j]*h[m] where m = j+k
    # = (flip(g_c) * h)[N-1-j] ... let me think more carefully.
    # Actually term2[j] = sum_{k=0}^{N-1-j} g_c[k]*h[j+k]
    # Let's define f[k] = g_c[N-1-k] for k=0..N-1
    # Then term2[j] = sum_{k=0}^{N-1-j} g_c[k]*h[j+k]
    # = sum_{m=j}^{N-1} g_c[m-j]*h[m]
    # This is the cross-correlation of g_c and h evaluated at lag (N-1-j) ... 
    # Actually, let's just compute it directly with a different FFT trick.
    
    # Alternative: term2[j] = sum_{k} g_c[k] * h[j+k] * [0<=j+k<N]
    # = sum_{m} g_c[m-j] * h[m] * [0<=m<N, 0<=m-j<N]
    # = sum_{m} g_c[m-j] * h[m] * [j<=m<N]
    
    # Let's use the fact that if we reverse g_c:
    # rev_g[k] = g_c[N-1-k]
    # Then (rev_g * h)[N-1-j] = sum_k rev_g[k]*h[N-1-j-k] = sum_k g_c[N-1-k]*h[N-1-j-k]
    # Let m = N-1-k, so k = N-1-m:
    # = sum_m g_c[m]*h[N-1-j-(N-1-m)] = sum_m g_c[m]*h[m-j]
    # That's not quite right either.
    
    # Let me just compute term2 using a simple approach with numpy operations:
    # For each j, term2[j] = dot(g_c[:N-j], h[j:])
    # This is O(N^2) in the worst case but N=100 so it's 10000 ops which is fine.
    # But let's try to vectorize it better.
    
    # Actually, for N=100, the direct computation is fast enough. The bottleneck
    # was the Python loop over k in the original. Let's use a matrix approach.
    
    # Build the matrix M where M[k,j] = h[j-k] if j>=k else 0, plus h[j+k] if j+k<N else 0
    # Then g_h = -(2/N) * M^T @ g_c
    
    # Actually, let's just do it with the FFT convolution for term1 and a direct
    # computation for term2, or better yet, use the full matrix for N=100.
    
    # For N=100, building the 100x100 matrix is trivial.
    j_idx = np.arange(N)
    k_idx = np.arange(N)
    J, K = np.meshgrid(j_idx, k_idx, indexing='ij')
    
    # term1_matrix[k,j] = h[j-k] if j>=k else 0
    mask1 = J >= K
    term1_mat = np.where(mask1, h[J - K], 0.0)
    
    # term2_matrix[k,j] = h[j+k] if j+k<N else 0
    mask2 = J + K < N
    term2_mat = np.where(mask2, h[J + K], 0.0)
    
    # g_h[j] = -(2/N) * sum_k g_c[k] * (term1_mat[k,j] + term2_mat[k,j])
    g_h = -(2.0 / N) * (term1_mat.T @ g_c + term2_mat.T @ g_c)
    
    return val, g_h


def _projected_gradient_step(h, grad, lr):
    """One projected gradient step: h_new = proj(h - lr * grad)."""
    h_new = h - lr * grad
    return _project(h_new)


def _nesterov_accelerated_gradient(h0, beta, n_steps=3000, lr=None, time_budget=10.0):
    """Nesterov accelerated projected gradient on the smooth surrogate.
    Returns best h found and its score."""
    t_start = time.time()
    h = h0.copy()
    y = h.copy()
    best_h = h.copy()
    best_val = _smooth_obj_and_grad(h, beta)[0]
    
    if lr is None:
        lr = 1.0 / (beta * 2.0)
    
    momentum = 0.9
    
    for step in range(n_steps):
        if time.time() - t_start > time_budget:
            break
        
        val_y, grad_y = _smooth_obj_and_grad(y, beta)
        
        h_new = y - lr * grad_y
        h_new = _project(h_new)
        
        t_k = (1.0 + np.sqrt(1.0 + 4.0 * momentum ** 2)) / 2.0
        y = h_new + (momentum / t_k) * (h_new - h)
        y = _project(y)
        h = h_new
        
        val_h, _ = _smooth_obj_and_grad(h, beta)[0]
        if val_h < best_val:
            best_val = val_h
            best_h = h.copy()
        
        if step > 10 and val_h > best_val + 1e-10:
            lr *= 0.95
    
    return best_h, _score(best_h)


def _refine_smooth(h0, beta, n_max_iter=3000, time_budget=25.0):
    """Refine h using Nesterov accelerated projected gradient on the smooth surrogate."""
    h0 = _project(h0.copy())
    h_opt, s_opt = _nesterov_accelerated_gradient(h0, beta, n_steps=n_max_iter, time_budget=time_budget)
    return h_opt, s_opt


def _random_start(rng):
    """Random feasible start."""
    h = rng.uniform(0.0, 1.0, N)
    return _project(h)


def _perturb(h, rng, scale=0.05):
    """Add small random perturbation and re-project."""
    h_p = h + rng.normal(0.0, scale, N)
    return _project(h_p)


def _construct_direct():
    """Direct multi-frequency construction to flatten autocorrelation."""
    i = np.arange(N)
    h = np.full(N, 0.5)
    freqs = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]
    amplitudes = [0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
    phases = [0.0, 0.7, 1.3, 2.1, 0.4, 1.8, 2.5, 0.9, 1.6, 0.3]
    for f, a, p in zip(freqs, amplitudes, phases):
        h += a * np.cos(2.0 * np.pi * f * i / N + p)
    return _project(h)


def _high_beta_polish(h0, time_budget=10.0):
    """Final high-precision polish with very high beta and careful LR scheduling."""
    t_start = time.time()
    h = h0.copy()
    best_h = h.copy()
    best_score = _score(h)
    
    # Use very high beta for near-exact max approximation
    beta = 1000.0
    
    # Start with small LR and gradually increase
    lr = 0.01
    momentum = 0.9
    y = h.copy()
    
    n_steps = 5000
    for step in range(n_steps):
        if time.time() - t_start > time_budget:
            break
        
        val_y, grad_y = _smooth_obj_and_grad(y, beta)
        
        h_new = y - lr * grad_y
        h_new = _project(h_new)
        
        t_k = (1.0 + np.sqrt(1.0 + 4.0 * momentum ** 2)) / 2.0
        y = h_new + (momentum / t_k) * (h_new - h)
        y = _project(y)
        h = h_new
        
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
        
        # Adaptive LR
        if step > 20 and s > best_score + 1e-12:
            lr *= 0.9
        elif step % 100 == 0 and step > 100:
            # Occasionally try a larger step
            lr = min(lr * 1.1, 0.1)
    
    return best_h, best_score


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

    # Phase 2: Diverse frequency-based starts
    if time.time() - t0 < _DEADLINE - 60:
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
            if time.time() - t0 > _DEADLINE - 60:
                break
            h = np.full(N, 0.5)
            for f, a in zip(freqs, amps):
                h += a * np.cos(2.0 * np.pi * f * i / N)
            _update(h)

    # Phase 3: Random starts
    if time.time() - t0 < _DEADLINE - 50:
        for _ in range(8):
            if time.time() - t0 > _DEADLINE - 50:
                break
            h_rand = _random_start(rng)
            _update(h_rand)

    # Phase 4: Smooth surrogate refinement with Nesterov accelerated gradient
    # Use annealed beta for sharper max approximation
    beta_list = [20.0, 50.0, 100.0, 200.0, 400.0]
    
    for beta in beta_list:
        remaining = _DEADLINE - time.time()
        if remaining < 15:
            break
        
        time_per_beta = min(18.0, (remaining - 10.0) / 2.0)
        if time_per_beta < 5.0:
            break
        
        # Full-space refinement
        h_ref, s_ref = _refine_smooth(best_h, beta=beta, n_max_iter=4000, time_budget=time_per_beta / 2.0)
        _update(h_ref)
        
        # Add perturbation of best for diversity
        if best_h is not None and time.time() - t0 < _DEADLINE - 20:
            h_pert = _perturb(best_h, rng, scale=0.02)
            h_ref2, s_ref2 = _refine_smooth(h_pert, beta=beta, n_max_iter=2500, 
                                            time_budget=min(10.0, _DEADLINE - time.time() - 10))
            _update(h_ref2)

    # Phase 5: High-precision polish with very high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 15:
        remaining = _DEADLINE - time.time() - 5.0
        if remaining > 10.0:
            h_polish, s_polish = _high_beta_polish(best_h, time_budget=min(12.0, remaining))
            _update(h_polish)
        
        # Another round of perturbation + polish
        if best_h is not None and time.time() - t0 < _DEADLINE - 10:
            remaining2 = _DEADLINE - time.time() - 5.0
            if remaining2 > 8.0:
                h_pert = _perturb(best_h, rng, scale=0.01)
                h_polish2, s_polish2 = _high_beta_polish(h_pert, time_budget=min(10.0, remaining2))
                _update(h_polish2)

    # Phase 6: Local perturbation search around best
    if best_h is not None and time.time() - t0 < _DEADLINE - 12:
        for scale in [0.015, 0.008, 0.004]:
            if time.time() - t0 > _DEADLINE - 12:
                break
            for _ in range(4):
                if time.time() - t0 > _DEADLINE - 12:
                    break
                h_pert = _perturb(best_h, rng, scale=scale)
                _update(h_pert)
            if best_h is not None:
                remaining = _DEADLINE - time.time() - 6.0
                if remaining > 5.0:
                    h_ref, s_ref = _refine_smooth(best_h, beta=500.0, n_max_iter=1500,
                                                  time_budget=min(6.0, remaining))
                    _update(h_ref)

    if best_h is None:
        best_h = np.full(N, 0.5)

    return _project(best_h)
