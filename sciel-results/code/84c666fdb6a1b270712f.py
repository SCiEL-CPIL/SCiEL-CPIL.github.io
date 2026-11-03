"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses SDP relaxation insights with custom interior-point-inspired solver.
"""
import time
import numpy as np

N = 100
_DEADLINE = time.time() + 115


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _project(h, target_sum):
    """
    Project h onto {x | sum(x) = target_sum, 0 <= x_i <= 1}.
    """
    h = np.asarray(h, dtype=np.float64)
    
    def sum_proj(tau):
        return np.sum(np.clip(h + tau, 0.0, 1.0))
    
    lo = -max(np.max(h), 1.0) - 1.0
    hi = 1.0 - min(np.min(h), 0.0) + 1.0
    
    if sum_proj(lo) > target_sum:
        lo -= 10.0
    if sum_proj(hi) < target_sum:
        hi += 10.0
        
    for _ in range(50):
        mid = (lo + hi) / 2.0
        s = sum_proj(mid)
        if s < target_sum:
            lo = mid
        else:
            hi = mid
            
    tau = (lo + hi) / 2.0
    h_proj = np.clip(h + tau, 0.0, 1.0)
    
    diff = target_sum - np.sum(h_proj)
    if abs(diff) > 1e-12:
        mask = (h_proj > 1e-9) & (h_proj < 1.0 - 1e-9)
        if np.any(mask):
            h_proj[mask] += diff / np.sum(mask)
            h_proj = np.clip(h_proj, 0.0, 1.0)
            
    return h_proj


def _lowpass_filter(h, cutoff_freq):
    """
    Apply a lowpass filter to h to encourage smoothness (SDP-like behavior).
    """
    H = np.fft.rfft(h)
    n_freqs = len(H)
    # Keep frequencies up to cutoff_freq
    mask = np.zeros(n_freqs)
    mask[:cutoff_freq + 1] = 1.0
    # Smooth the cutoff slightly
    if cutoff_freq + 1 < n_freqs:
        mask[cutoff_freq + 1] = 0.5
    H_filtered = H * mask
    h_filtered = np.fft.irfft(H_filtered, n=len(h))
    return h_filtered


def _optimize(h0, max_iter=1000, beta_start=10.0, beta_end=1000.0):
    """
    Optimize using smooth max with adaptive step size and lowpass regularization.
    """
    h = _project(h0.copy(), N / 2.0)
    h_prev = h.copy()
    g_prev = None
    beta = beta_start
    best_local_score = _score(h)
    best_local_h = h.copy()
    
    # Add a small lowpass regularization to the gradient to mimic SDP smoothness
    reg_strength = 0.01
    
    for i in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break
            
        # Animate beta
        progress = i / max_iter
        beta = beta_start * (beta_end / beta_start) ** progress
        
        cs = _linear_overlap_values(h)
        m = np.max(cs)
        
        # Smooth max approximation
        exp_terms = np.exp(beta * (cs - m))
        sum_exp = np.sum(exp_terms)
        w = exp_terms / sum_exp
        val = m + np.log(sum_exp) / beta
        
        grad = np.zeros(N)
        for k in range(N):
            if w[k] < 1e-9:
                continue
            factor = -(2.0 / N) * w[k]
            if k > 0:
                grad[k:] += factor * h[:N-k]
            if k < N:
                grad[:N-k] += factor * h[k:]
                
        # Add lowpass regularization gradient
        # Gradient of ||(I - P_lowpass) h||^2
        h_lp = _lowpass_filter(h, 5)
        grad_reg = 2 * reg_strength * (h - h_lp)
        grad += grad_reg

        # Adaptive Step Size
        if g_prev is not None:
            s_vec = h - h_prev
            y_vec = grad - g_prev
            s_norm_sq = np.dot(s_vec, s_vec)
            y_norm_sq = np.dot(y_vec, y_vec)
            sy = np.dot(s_vec, y_vec)
            
            if s_norm_sq > 1e-12 and y_norm_sq > 1e-12 and sy > 1e-12:
                alpha_bb1 = s_norm_sq / sy
                alpha_bb2 = sy / y_norm_sq
                alpha = 0.5 * (alpha_bb1 + alpha_bb2)
                alpha = np.clip(alpha, 1e-4, 1.0)
            else:
                alpha = 0.01
        else:
            alpha = 0.01

        # Backtracking Line Search
        h_new = h - alpha * grad
        h_new = _project(h_new, N / 2.0)
        
        score_new = _score(h_new)
        score_curr = _score(h)
        
        if score_new > score_curr:
            # If it gets worse, reduce alpha
            for _ in range(10):
                alpha *= 0.5
                h_new = h - alpha * grad
                h_new = _project(h_new, N / 2.0)
                score_new = _score(h_new)
                if score_new <= score_curr:
                    break
            if score_new > score_curr:
                # Accept anyway if improvement is small, or skip?
                # To avoid stalling, accept if very close
                if score_new < score_curr + 1e-6:
                    pass
                else:
                    # Try random perturbation?
                    h_new = h + np.random.normal(0, 0.01, N)
                    h_new = _project(h_new, N / 2.0)
                    score_new = _score(h_new)

        h_prev = h
        h = h_new
        g_prev = grad
        
        if score_new < best_local_score:
            best_local_score = score_new
            best_local_h = h.copy()
            
        # Convergence check
        if np.linalg.norm(h - h_prev) < 1e-9:
            break
            
    return best_local_h


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    target_sum = N / 2.0

    def _update_best(h):
        nonlocal best_h, best_score
        h = _project(h, target_sum)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    # 1. Smooth (Low-Frequency) Starts - Mimicking SDP PSD constraint
    i = np.arange(N)
    for freq in [1, 2, 3, 4, 5]:
        if time.time() - t0 > _DEADLINE - 20:
            break
        # Cosine starts
        h_cos = 0.5 + 0.4 * np.cos(2.0 * np.pi * freq * i / N)
        h_opt = _optimize(h_cos)
        _update_best(h_opt)
        
        # Sine starts
        h_sin = 0.5 + 0.4 * np.sin(2.0 * np.pi * freq * i / N)
        h_opt = _optimize(h_sin)
        _update_best(h_opt)

    # 2. Block/Step Starts
    if time.time() - t0 < _DEADLINE - 20:
        for width in [10, 20, 25, 50]:
            h_block = np.zeros(N)
            h_block[:width] = 1.0
            h_opt = _optimize(h_block)
            _update_best(h_opt)
            
            # Shifted blocks
            h_block_shift = np.roll(h_block, width // 2)
            h_opt = _optimize(h_block_shift)
            _update_best(h_opt)

    # 3. Random Starts with Lowpass Filtering
    if time.time() - t0 < _DEADLINE - 10:
        for _ in range(5):
            if time.time() - t0 > _DEADLINE - 5:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _lowpass_filter(h_rand, 3) # Smooth it out
            h_opt = _optimize(h_rand)
            _update_best(h_opt)

    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _project(best_h, target_sum)
    return best_h
