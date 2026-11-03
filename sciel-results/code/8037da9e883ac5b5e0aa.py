import time
import numpy as np

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill

def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=0}^{N-1-k} h_i h_{i+k})
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

def _logsumexp_obj_and_grad(h, beta):
    """
    Compute the LogSumExp smoothed maximum of C_k and its gradient.
    Objective: F(h) = (1/beta) * log(sum(exp(beta * C_k)))
    Gradient: sum(w_k * grad_C_k)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for the soft max
    
    val = m + np.log(sum_exp) / beta
    
    grad = np.zeros(n)
    
    # dC_k/dh_j = -(2/n) * (h_{j-k} if j>=k + h_{j+k} if j+k<n)
    for k in range(n):
        if w[k] == 0.0:
            continue
        factor = -(2.0 / n) * w[k]
        
        if k > 0:
            grad[k:] += factor * h[:n-k]
        if k < n:
            grad[:n-k] += factor * h[k:]
            
    return val, grad

def _project_sum_box(y, target_sum):
    """
    Project y onto {x : 0 <= x_i <= 1, sum(x) = target_sum}
    """
    lo = np.min(y) - 1.0
    hi = np.max(y) + 1.0
    
    def g(tau):
        x = np.clip(y + tau, 0.0, 1.0)
        return np.sum(x) - target_sum
    
    if g(lo) > 0:
        return np.clip(y + lo, 0.0, 1.0)
    if g(hi) < 0:
        return np.clip(y + hi, 0.0, 1.0)
        
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if g(mid) > 0:
            hi = mid
        else:
            lo = mid
            
    tau = (lo + hi) / 2.0
    return np.clip(y + tau, 0.0, 1.0)

def _nesterov_pgd(h0, beta, max_iter=500, lr=0.05, mu=0.9):
    """
    Nesterov-accelerated Projected Gradient Descent.
    """
    if time.time() > _DEADLINE - 5:
        return h0

    h = np.array(h0, dtype=np.float64)
    y = h.copy()
    t_k = 1.0
    
    for i in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
            
        # Nesterov step
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_k**2)) / 2.0
        beta_k = (t_k - 1.0) / t_next
        
        y = h + beta_k * (h - y)
        
        val, grad = _logsumexp_obj_and_grad(y, beta)
        
        # Gradient step
        h_new = y - lr * grad
        
        # Project
        h = _project_sum_box(h_new, N / 2.0)
        
        t_k = t_next
        
    return h

def _coordinate_polish(h, max_iter=2000):
    """
    Coordinate descent polish on the active set (maximizing overlap reduction).
    This is a simple heuristic to shave off residual error.
    """
    h = np.array(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    max_c = np.max(cs)
    
    for _ in range(max_iter):
        if time.time() > _DEADLINE - 2:
            break
            
        # Find the index j that gives the best improvement if we change h_j slightly
        # This is expensive if done naively. Let's just do a simple local search.
        # Actually, let's just return h if time is tight. The PGD should be good enough.
        # A full coordinate polish might be too slow for 100 vars if not optimized.
        # Let's try a simple random perturbation and accept if better.
        pass
    return h

def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        # Ensure strict feasibility
        h = _project_sum_box(h, N / 2.0)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
            
    # 1. Initial Candidates
    i = np.arange(N)
    
    # Cosine wave
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _project_sum_box(h_cos, N / 2.0)
    _update_best(h_cos)
    
    # Sine wave
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    h_sin = _project_sum_box(h_sin, N / 2.0)
    _update_best(h_sin)
    
    # Random starts
    starts = [h_cos, h_sin]
    for _ in range(5):
        h_rand = np.random.uniform(0, 1, N)
        h_rand = _project_sum_box(h_rand, N / 2.0)
        starts.append(h_rand)
        _update_best(h_rand)
        
    # 2. Optimization Phases using Nesterov PGD
    # Strategy: Start with low beta (smooth), then increase beta (sharper max)
    
    # Phase 1: Coarse optimization
    if best_h is not None and time.time() - t0 < _DEADLINE - 60:
        h_opt = _nesterov_pgd(best_h, beta=10.0, max_iter=500, lr=0.02, mu=0.9)
        _update_best(h_opt)
        
    # Phase 2: Medium beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 30:
        h_opt = _nesterov_pgd(best_h, beta=50.0, max_iter=1000, lr=0.01, mu=0.95)
        _update_best(h_opt)
        
    # Phase 3: High beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 15:
        h_opt = _nesterov_pgd(best_h, beta=200.0, max_iter=2000, lr=0.005, mu=0.98)
        _update_best(h_opt)
        
    # Phase 4: Ultra-high beta polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        h_opt = _nesterov_pgd(best_h, beta=1000.0, max_iter=5000, lr=0.001, mu=0.99)
        _update_best(h_opt)

    # If we still have time, try refining a different start
    if time.time() - t0 < _DEADLINE - 20:
        # Pick a random start that wasn't the best
        h_alt = np.random.uniform(0, 1, N)
        h_alt = _project_sum_box(h_alt, N / 2.0)
        h_opt = _nesterov_pgd(h_alt, beta=50.0, max_iter=1000, lr=0.01, mu=0.95)
        _update_best(h_opt)
        if time.time() - t0 < _DEADLINE - 10:
            h_opt = _nesterov_pgd(h_opt, beta=200.0, max_iter=2000, lr=0.005, mu=0.98)
            _update_best(h_opt)

    if best_h is None:
        best_h = np.full(N, 0.5)
        
    # Final projection to ensure exact sum and bounds
    best_h = _project_sum_box(best_h, N / 2.0)
    
    return best_h
