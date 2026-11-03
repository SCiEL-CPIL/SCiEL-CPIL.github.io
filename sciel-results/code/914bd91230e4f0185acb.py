"""Erdos minimum overlap: smooth-surrogate optimization with bisection on t;
uses numpy, time; defines _autocorr, _overlap_values, _score, _project,
_smooth_max_and_grad, _smooth_obj_and_grad, _projected_gradient_step,
_nesterov_accelerated_gradient, _bisection_check, _refine_smooth,
_construct_direct, _random_start, construct."""
import time
import numpy as np

N = 100
_TARGET = N / 2.0
_NFFT = 256  # next power of 2 >= 2*N
_DEADLINE = time.time() + 145  # Safety margin before 150s kill


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


def _smooth_obj_and_grad(h, beta):
    """f(h) = S_beta(C(h)); returns (f, df/dh) using vectorized computation.
    
    C_k = (2/N)*(N/2 - r_k) where r_k = autocorr[h][k]
    dC_k/dh_i = -(2/N) * (delta_{i>=k} * h_{i-k} + delta_{i<k} * h_{i+k}) for k>0
               -(2/N) * 2*h_i for k=0
    
    g_h = factor * sum_k g_c[k] * dC_k/dh_i
    """
    cs = _overlap_values(h)
    val, g_c = _smooth_max_and_grad(cs, beta)
    n = N
    factor = -(2.0 / n)
    
    # Vectorized gradient computation using meshgrid
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


def _projected_gradient_step(h, grad, lr):
    """Single projected gradient step."""
    h_new = h - lr * grad
    return _project(h_new)


def _nesterov_accelerated_gradient(h0, beta, n_steps=5000, lr=None, time_budget=10.0):
    """Nesterov accelerated projected gradient on the smooth surrogate."""
    t_start = time.time()
    h = _project(h0.copy())
    y = h.copy()
    best_h = h.copy()
    best_val = _smooth_obj_and_grad(h, beta)[0]

    if lr is None:
        lr = 1.0 / (beta * 12.0)

    t_k = 1.0

    for step in range(n_steps):
        if time.time() - t_start > time_budget:
            break

        val_y, grad_y = _smooth_obj_and_grad(y, beta)
        h_new = _projected_gradient_step(y, grad_y, lr)
        val_new = _smooth_obj_and_grad(h_new, beta)[0]

        # Backtracking
        for _ in range(8):
            if val_new <= best_val + 1e-13:
                break
            lr *= 0.7
            h_new = _projected_gradient_step(y, grad_y, lr)
            val_new = _smooth_obj_and_grad(h_new, beta)[0]
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


def _bisection_check(t_val, h0, beta, n_steps=2000, time_budget=5.0):
    """Check if max overlap <= t_val is achievable by minimizing the epigraph surrogate.
    Minimize max(0, S_beta(C(h)) - t_val) subject to constraints.
    Returns (h, achieved_max) if feasible, or (None, None) if not.
    """
    def _epigraph_obj(h):
        cs = _overlap_values(h)
        val, _ = _smooth_max_and_grad(cs, beta)
        return max(0.0, val - t_val)
    
    def _epigraph_grad(h):
        cs = _overlap_values(h)
        val, g_c = _smooth_max_and_grad(cs, beta)
        if val <= t_val:
            return np.zeros(N)
        # Gradient of (val - t_val) w.r.t. h
        n = N
        factor = -(2.0 / n)
        i_idx = np.arange(n)
        k_idx = np.arange(n)
        I, K = np.meshgrid(i_idx, k_idx, indexing='ij')
        fwd_mask = (I >= K) & (K >= 1)
        fwd_vals = np.where(fwd_mask, g_c[K] * h[np.where(fwd_mask, I - K, 0)], 0.0)
        fwd_sum = np.sum(fwd_vals, axis=1)
        bwd_mask = (I + K < n) & (K >= 1)
        bwd_vals = np.where(bwd_mask, g_c[K] * h[np.where(bwd_mask, I + K, 0)], 0.0)
        bwd_sum = np.sum(bwd_vals, axis=1)
        g_h = factor * (g_c[0] * 2.0 * h + fwd_sum + bwd_sum)
        return g_h
    
    # Simple projected gradient descent on the epigraph objective
    t_start = time.time()
    h = _project(h0.copy())
    best_h = h.copy()
    best_obj = _epigraph_obj(h)
    lr = 0.01 / (beta ** 0.5)
    
    for step in range(n_steps):
        if time.time() - t_start > time_budget:
            break
        obj = _epigraph_obj(h)
        if obj < 1e-10:
            return h, _score(h)
        grad = _epigraph_grad(h)
        h_new = _projected_gradient_step(h, grad, lr)
        obj_new = _epigraph_obj(h_new)
        if obj_new > obj:
            lr *= 0.7
            h_new = _projected_gradient_step(h, grad, lr)
            obj_new = _epigraph_obj(h_new)
        h = h_new
        if obj_new < best_obj:
            best_obj = obj_new
            best_h = h.copy()
    
    final_score = _score(best_h)
    if final_score <= t_val + 1e-8:
        return best_h, final_score
    return None, None


def _refine_smooth(h0, beta, n_steps=3000, lr=None, time_budget=8.0):
    """Refine using smooth surrogate with given beta."""
    return _nesterov_accelerated_gradient(h0, beta, n_steps=n_steps, lr=lr, time_budget=time_budget)


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
    beta_list = [20.0, 100.0, 300.0, 800.0, 2000.0, 5000.0]

    for beta in beta_list:
        remaining = _DEADLINE - time.time()
        if remaining < 12:
            break

        time_per_beta = min(15.0, (remaining - 8.0) / 1.5)
        if time_per_beta < 4.0:
            break

        h_ref, s_ref = _nesterov_accelerated_gradient(best_h, beta=beta, n_steps=8000,
                                                       time_budget=time_per_beta)
        _update(h_ref)

        # Perturbation restart for diversity
        if best_h is not None and time.time() - t0 < _DEADLINE - 15:
            h_pert = best_h + rng.normal(0.0, 0.015, N)
            h_pert = _project(h_pert)
            h_ref2, s_ref2 = _nesterov_accelerated_gradient(h_pert, beta=beta, n_steps=4000,
                                                             time_budget=min(6.0, _DEADLINE - time.time() - 8))
            _update(h_ref2)

    # Phase 5: Final high-precision polish with very high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        remaining = _DEADLINE - time.time() - 2.0
        if remaining > 3.0:
            h_polish, s_polish = _nesterov_accelerated_gradient(best_h, beta=10000.0, n_steps=5000,
                                                                 time_budget=min(6.0, remaining))
            _update(h_polish)

            if time.time() - t0 < _DEADLINE - 2:
                remaining2 = _DEADLINE - time.time() - 1.0
                if remaining2 > 1.5:
                    h_polish2, s_polish2 = _nesterov_accelerated_gradient(best_h, beta=20000.0, n_steps=2000,
                                                                           time_budget=min(3.0, remaining2))
                    _update(h_polish2)

    if best_h is None:
        best_h = np.full(N, 0.5)

    return _project(best_h)
