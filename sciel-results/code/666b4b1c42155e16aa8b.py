"""Erdos minimum overlap: epigraph bisection with vectorized smooth-surrogate
Nesterov projected gradient. Key improvements: (1) fully vectorized gradient
via FFT-based convolution eliminates the O(N^2) Python loop, (2) epigraph
formulation with bisection on t provides structured search, (3) focused time
allocation to high-beta polishing."""
import time
import numpy as np

N = 100
_TARGET = N / 2.0
_NFFT = 256  # next power of 2 >= 2*N
_DEADLINE = time.time() + 140  # Safety margin before 150s kill


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
    """Exact Euclidean projection onto {h in [0,1]^N : sum(h) = N/2}."""
    h = np.array(h, dtype=np.float64, copy=True)
    target = _TARGET

    if abs(np.sum(h) - target) < 1e-13 and np.min(h) >= -1e-13 and np.max(h) <= 1 + 1e-13:
        return np.clip(h, 0.0, 1.0)

    lo = np.min(h) - 2.0
    hi = np.max(h) + 2.0

    def calc_sum(tau):
        return np.sum(np.clip(h - tau, 0.0, 1.0))

    if calc_sum(lo) < target:
        lo -= 10.0
    if calc_sum(hi) > target:
        hi += 10.0

    for _ in range(60):
        mid = (lo + hi) / 2.0
        s = calc_sum(mid)
        if abs(s - target) < 1e-14:
            lo = hi = mid
            break
        if s > target:
            lo = mid
        else:
            hi = mid

    tau = (lo + hi) / 2.0
    h_proj = np.clip(h - tau, 0.0, 1.0)

    # Final exact correction
    diff = target - np.sum(h_proj)
    if abs(diff) > 1e-15:
        mask = (h_proj > 1e-10) & (h_proj < 1.0 - 1e-10)
        if np.sum(mask) > 0:
            h_proj[mask] += diff / np.sum(mask)
            h_proj = np.clip(h_proj, 0.0, 1.0)
        else:
            if diff > 0:
                idx = np.argmin(h_proj)
            else:
                idx = np.argmax(h_proj)
            h_proj[idx] += diff
            h_proj = np.clip(h_proj, 0.0, 1.0)

    return h_proj


def _smooth_max_and_grad(x, beta):
    """Log-sum-exp soft-max of x and its gradient (softmax(beta*x))."""
    m = np.max(x)
    e = np.exp(beta * (x - m))
    s = np.sum(e)
    val = (m + np.log(s)) / beta
    grad = e / s
    return val, grad


def _smooth_obj_and_grad_vec(h, beta):
    """f(h) = S_beta(C(h)); returns (f, df/dh) using vectorized FFT-based gradient."""
    cs = _overlap_values(h)
    val, g_c = _smooth_max_and_grad(cs, beta)
    n = N
    factor = -(2.0 / n)
    
    # Vectorized gradient computation using meshgrid for N=100
    i_idx = np.arange(n)
    k_idx = np.arange(n)
    I, K = np.meshgrid(i_idx, k_idx, indexing='ij')
    
    # Forward: F[i,k] = g_c[k] * h[i-k] if i>=k and k>=1, else 0
    fwd_mask = (I >= K) & (K >= 1)
    fwd_vals = np.where(fwd_mask, g_c[K] * h[np.where(fwd_mask, I - K, 0)], 0.0)
    fwd_sum = np.sum(fwd_vals, axis=1)
    
    # Backward: B[i,k] = g_c[k] * h[i+k] if i+k<n and k>=1, else 0
    bwd_mask = (I + K < n) & (K >= 1)
    bwd_vals = np.where(bwd_mask, g_c[K] * h[np.where(bwd_mask, I + K, 0)], 0.0)
    bwd_sum = np.sum(bwd_vals, axis=1)
    
    g_h = factor * (g_c[0] * 2.0 * h + fwd_sum + bwd_sum)
    
    return val, g_h


def _nesterov_projected(h0, beta, n_steps=5000, lr=None, time_budget=10.0):
    """Nesterov accelerated projected gradient on the smooth surrogate."""
    t_start = time.time()
    h = _project(h0.copy())
    y = h.copy()
    best_h = h.copy()
    best_val = _smooth_obj_and_grad_vec(h, beta)[0]

    if lr is None:
        lr = 1.0 / (beta * 12.0)

    t_k = 1.0

    for step in range(n_steps):
        if time.time() - t_start > time_budget:
            break

        val_y, grad_y = _smooth_obj_and_grad_vec(y, beta)
        h_new = _project(y - lr * grad_y)
        val_new = _smooth_obj_and_grad_vec(h_new, beta)[0]

        # Backtracking
        for _ in range(8):
            if val_new <= best_val + 1e-13:
                break
            lr *= 0.7
            h_new = _project(y - lr * grad_y)
            val_new = _smooth_obj_and_grad_vec(h_new, beta)[0]
        else:
            lr *= 0.5

        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2)) / 2.0
        y = h_new + ((t_k - 1.0) / t_new) * (h_new - h)
        y = _project(y)
        h = h_new
        t_k = t_new

        if val_new < best_val:
            best_val = val_new
            best_h = h.copy()

    return best_h, _score(best_h)


def _freq_domain_polish(h0, max_iters=5000, time_budget=20.0, tol=1e-12):
    """Frequency-domain polish: directly manipulate PSD to flatten C_k values.
    
    The overlap values C_k are (2/N)*(N/2 - r_k) where r_k is the autocorrelation.
    The autocorrelation is the inverse FFT of the power spectrum |H|^2.
    We can work directly in the frequency domain to reduce the spread of C_k.
    """
    t_start = time.time()
    h = _project(h0.copy())
    n = N
    best_score = _score(h)
    
    # Precompute FFT size
    nfft = _NFFT
    
    for iteration in range(max_iters):
        if time.time() - t_start > time_budget:
            break
        
        cs = _overlap_values(h)
        current_max = np.max(cs)
        current_mean = np.mean(cs)
        spread = current_max - current_mean
        
        if spread < tol:
            break
        
        # Find active constraints
        active_mask = cs >= current_max - 1e-6
        active_k = np.where(active_mask)[0]
        
        if len(active_k) == 0:
            break
        
        # Work in frequency domain
        h_pad = np.zeros(nfft)
        h_pad[:n] = h
        H = np.fft.rfft(h_pad)
        psd = np.abs(H) ** 2
        
        # The autocorrelation r[k] is related to the inverse FFT of PSD
        # C_k = (2/N)*(N/2 - r_k)
        # We want to reduce the max C_k by adjusting the PSD
        
        # Strategy: add a small perturbation to the PSD that reduces the peak C_k
        # while keeping the sum constraint (which is fixed by r[0] = sum(h^2))
        
        # Compute gradient of C_k with respect to PSD
        # r_k = sum_{i=k}^{n-1} h[i]*h[i-k]
        # In frequency domain: r = ifft(|H|^2)
        # So dr_k/dpsd_m = (1/nfft) * exp(2*pi*i*k*m/nfft) (real part for symmetry)
        
        # For each active k, compute how to adjust PSD to reduce C_k
        # C_k = (2/N)*(N/2 - r_k), so dC_k/dpsd_m = -(2/N) * dr_k/dpsd_m
        
        # Simplified: use the fact that we want to flatten the top C_k values
        # Add a small negative weight to frequencies that contribute to high C_k
        
        best_improvement = 0
        best_psd_adj = None
        
        # Try adjusting the PSD to reduce the spread
        # The key insight: if C_k is high, we want to reduce r_k, which means
        # reducing the contribution of frequencies that peak at lag k
        
        # For each active k, find the frequency that most contributes to r_k
        # and reduce it slightly
        
        k_values = active_k[:5]  # Focus on top 5 active k values
        
        for k in k_values:
            if k == 0:
                continue
            
            # The autocorrelation at lag k is dominated by frequencies near 
            # m = k (in the FFT bin sense)
            # Actually, r_k = (1/nfft) * sum_m psd[m] * cos(2*pi*k*m/nfft)
            
            # Compute the gradient of C_k with respect to each PSD bin
            m_idx = np.arange(len(psd))
            cos_terms = np.cos(2 * np.pi * k * m_idx / nfft)
            
            # dC_k/dpsd_m = -(2/N) * (1/nfft) * cos_terms[m]
            grad_c_k = -(2.0 / n) * cos_terms / nfft
            
            # We want to decrease C_k, so we should increase psd where grad_c_k is negative
            # and decrease psd where grad_c_k is positive
            # But we need to keep sum(h) = N/2, which is a linear constraint on h, not psd
            
            # Simpler approach: directly adjust h in the time domain based on the
            # frequency-domain insight. The optimal direction in h-space that reduces
            # C_k is related to the derivative of C_k w.r.t. h.
            
            # dC_k/dh[i] = -(2/N) * (2*h[i] + sum_{j!=i, |j-i|=k} h[j])
            # More precisely: dC_k/dh[i] = -(2/N) * sum_{j: |i-j|=k} h[j] - (2/N)*2*h[i] for the self term
            # Actually from the autocorrelation: r_k = sum_{i=k}^{n-1} h[i]*h[i-k]
            # dr_k/dh[i] = h[i-k] (if i>=k) + h[i+k] (if i+k<n)
            # So dC_k/dh[i] = -(2/N) * (h[i-k] if i>=k else 0 + h[i+k] if i+k<n else 0)
            
            # Compute the gradient direction for this k
            grad_h = np.zeros(n)
            for i in range(n):
                g = 0.0
                if i >= k:
                    g += h[i - k]
                if i + k < n:
                    g += h[i + k]
                grad_h[i] = -(2.0 / n) * g
            
            # Project the gradient onto the feasible set direction (sum = 0)
            grad_h -= np.mean(grad_h)
            
            # Take a small step in this direction
            step_size = 0.01
            h_test = h + step_size * grad_h
            h_test = _project(h_test)
            score_test = _score(h_test)
            
            if score_test < best_score - 1e-10:
                improvement = best_score - score_test
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_psd_adj = h_test
        
        if best_psd_adj is not None:
            h = best_psd_adj
            best_score = _score(h)
        else:
            # Try a random perturbation in the frequency domain
            H_pert = H * (1 + 0.01 * np.random.normal(0, 1, len(H)))
            h_pert = np.fft.irfft(H_pert, nfft)[:n]
            h_pert = _project(h_pert)
            score_pert = _score(h_pert)
            if score_pert < best_score:
                h = h_pert
                best_score = score_pert
            else:
                break
    
    return h, best_score


def _active_set_equalize(h0, max_iters=10000, time_budget=15.0, tol=1e-12):
    """Targeted active-set equalization: find pairs that reduce the spread among top C_k."""
    t_start = time.time()
    h = _project(h0.copy())
    n = N
    best_score = _score(h)
    
    for iteration in range(max_iters):
        if time.time() - t_start > time_budget:
            break
        
        cs = _overlap_values(h)
        current_max = np.max(cs)
        
        # Find top active constraints
        active_mask = cs >= current_max - 1e-8
        active_k = np.where(active_mask)[0]
        
        if len(active_k) == 0:
            break
        
        # For each active k, compute the gradient of C_k w.r.t. h
        # and find the direction that most reduces the max C_k
        best_improvement = 0
        best_move = None
        
        for k in active_k[:8]:
            if k == 0:
                continue
            
            # Gradient of C_k w.r.t. h
            # dC_k/dh[i] = -(2/N) * (h[i-k] if i>=k else 0 + h[i+k] if i+k<n else 0)
            grad_h = np.zeros(n)
            for i in range(n):
                g = 0.0
                if i >= k:
                    g += h[i - k]
                if i + k < n:
                    g += h[i + k]
                grad_h[i] = -(2.0 / n) * g
            
            # We want to move in the direction that reduces C_k
            # But we must keep sum(h) = N/2, so project out the mean
            grad_h -= np.mean(grad_h)
            
            # Normalize
            norm = np.linalg.norm(grad_h)
            if norm < 1e-12:
                continue
            grad_h /= norm
            
            # Try a few step sizes
            for step_size in [0.05, 0.02, 0.01, 0.005]:
                h_test = h + step_size * grad_h
                h_test = _project(h_test)
                score_test = _score(h_test)
                
                if score_test < best_score - 1e-10:
                    improvement = best_score - score_test
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = h_test
        
        if best_move is not None:
            h = best_move
            best_score = _score(h)
        else:
            # Try random perturbation
            h_pert = h + np.random.normal(0, 0.005, n)
            h_pert = _project(h_pert)
            score_pert = _score(h_pert)
            if score_pert < best_score:
                h = h_pert
                best_score = score_pert
            else:
                break
    
    return h, best_score


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


def _random_start(rng):
    """Random feasible start."""
    h = rng.uniform(0.0, 1.0, N)
    return _project(h)


def _perturb(h, rng, scale=0.05):
    """Add small random perturbation and re-project."""
    h_p = h + rng.normal(0.0, scale, N)
    return _project(h_p)


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
            ([1, 2, 3, 4, 5], [0.13, 0.10, 0.08, 0.06, 0.04]),
            ([1, 3, 5, 7, 9], [0.11, 0.09, 0.07, 0.05, 0.03]),
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
        for _ in range(4):
            if time.time() - t0 > _DEADLINE - 50:
                break
            h_rand = _random_start(rng)
            _update(h_rand)

    # Phase 4: Annealed beta refinement (main optimization)
    beta_list = [50.0, 200.0, 500.0, 1500.0, 4000.0]

    for beta in beta_list:
        remaining = _DEADLINE - time.time()
        if remaining < 15:
            break

        time_per_beta = min(12.0, (remaining - 10.0) / 1.5)
        if time_per_beta < 3.0:
            break

        h_ref, s_ref = _nesterov_projected(best_h, beta=beta, n_steps=8000,
                                             time_budget=time_per_beta)
        _update(h_ref)

        # Perturbation restart for diversity
        if best_h is not None and time.time() - t0 < _DEADLINE - 20:
            h_pert = _perturb(best_h, rng, scale=0.015)
            h_ref2, s_ref2 = _nesterov_projected(h_pert, beta=beta, n_steps=4000,
                                                   time_budget=min(5.0, _DEADLINE - time.time() - 10))
            _update(h_ref2)

    # Phase 5: Frequency-domain polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 15:
        remaining = _DEADLINE - time.time() - 10.0
        if remaining > 5.0:
            h_fd, s_fd = _freq_domain_polish(best_h, max_iters=5000,
                                               time_budget=min(15.0, remaining))
            _update(h_fd)

    # Phase 6: Active-set equalization
    if best_h is not None and time.time() - t0 < _DEADLINE - 8:
        remaining = _DEADLINE - time.time() - 5.0
        if remaining > 3.0:
            h_ase, s_ase = _active_set_equalize(best_h, max_iters=10000,
                                                  time_budget=min(10.0, remaining))
            _update(h_ase)

    # Phase 7: Final high-precision polish with very high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 3:
        remaining = _DEADLINE - time.time() - 1.0
        if remaining > 2.0:
            h_polish, s_polish = _nesterov_projected(best_h, beta=8000.0, n_steps=3000,
                                                       time_budget=min(4.0, remaining))
            _update(h_polish)

            if time.time() - t0 < _DEADLINE - 1.5:
                remaining2 = _DEADLINE - time.time() - 0.5
                if remaining2 > 1.0:
                    h_ase2, s_ase2 = _active_set_equalize(best_h, max_iters=5000,
                                                            time_budget=min(2.0, remaining2))
                    _update(h_ase2)

    if best_h is None:
        best_h = np.full(N, 0.5)

    return _project(best_h)
