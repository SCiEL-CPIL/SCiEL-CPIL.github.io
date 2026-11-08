import numpy as np
import time

N = 50
_start_time = time.time()
_TIMEOUT = 110.0  # Safety margin

def _score(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-15):
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    # Full linear autoconvolution
    conv = np.convolve(h, h, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return l2_sq / (l1 * conv_inf)

def _greedy_support(K):
    """Find a support of size K that minimizes the max convolution peak for uniform weights."""
    if K == 1:
        return [0]
    
    # Start with 0
    S = [0]
    h = np.zeros(N)
    h[0] = 1.0
    best_peak = np.max(np.convolve(h, h, mode='full'))
    
    for _ in range(K - 1):
        best_j = -1
        best_val = float('inf')
        
        for j in range(1, N):
            if j in S:
                continue
            
            # Try adding j
            h_trial = h.copy()
            h_trial[j] = 1.0
            conv = np.convolve(h_trial, h_trial, mode='full')
            peak = np.max(conv)
            
            # We want to minimize the peak, or maximize the ratio.
            # For uniform weights, ratio is K / (K * peak_uniform_scaled) ?
            # Actually, if we add one, the total mass increases.
            # Let's just minimize the peak of the convolution of the 0/1 vector.
            # This corresponds to minimizing the max overlap.
            
            if peak < best_val:
                best_val = peak
                best_j = j
        
        if best_j != -1:
            S.append(best_j)
            h[best_j] = 1.0
        else:
            break
            
    return S

def _optimize_weights(S, max_iters=2000):
    """Optimize weights on a fixed support S to maximize the score."""
    K = len(S)
    if K == 0:
        return np.zeros(N), 0.0
    
    # Initialize with uniform weights
    w = np.ones(K) / K
    h = np.zeros(N)
    h[S] = w
    
    best_score = _score(h)
    best_w = w.copy()
    
    step_size = 0.1
    for i in range(max_iters):
        if time.time() - _start_time > _TIMEOUT:
            break
            
        # Random perturbation
        idx = np.random.randint(K)
        w_trial = w.copy()
        w_trial[idx] += np.random.normal(0, step_size)
        
        # Ensure non-negative and normalize
        if w_trial[idx] < 0:
            w_trial[idx] = 0
        w_trial = np.maximum(w_trial, 0)
        sum_w = np.sum(w_trial)
        if sum_w < 1e-15:
            continue
        w_trial /= sum_w
        
        # Construct h and score
        h_trial = np.zeros(N)
        h_trial[S] = w_trial
        s_trial = _score(h_trial)
        
        if s_trial > best_score:
            w = w_trial
            best_score = s_trial
            best_w = w.copy()
            step_size = max(step_size * 1.02, 0.01)
        else:
            step_size = max(step_size * 0.95, 0.001)
            
    h_best = np.zeros(N)
    h_best[S] = best_w
    return h_best, best_score

def construct():
    """Construct 50 nonnegative heights maximizing the ratio."""
    global _start_time
    _start_time = time.time()
    
    best_h = np.ones(N) / N
    best_score = _score(best_h)
    
    # Try different support sizes
    # K=1 gives 1.0
    # K=2 gives ~1.0
    # K=3 gives ~1.5
    # K=4 gives ~2.0
    # ...
    
    for K in range(2, 16):
        if time.time() - _start_time > _TIMEOUT:
            break
        
        S = _greedy_support(K)
        if len(S) < K:
            continue
            
        h_opt, s_opt = _optimize_weights(S, max_iters=1000)
        
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt
            
    # Final polish on the best found
    # Re-extract support from best_h? Or just polish weights on the best S?
    # We don't store the best S explicitly in the loop above, but we can just do a final random polish on best_h
    
    # Simple final polish
    for _ in range(500):
        if time.time() - _start_time > _TIMEOUT:
            break
        h_trial = best_h + np.random.normal(0, 0.01, size=N)
        h_trial = np.maximum(h_trial, 0)
        s_trial = _score(h_trial)
        if s_trial > best_score:
            best_score = s_trial
            best_h = h_trial
            
    return list(best_h)
