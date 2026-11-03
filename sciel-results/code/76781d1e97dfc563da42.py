"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Approach: Direct optimization using smooth Log-Sum-Exp approximation of the max,
with Nesterov accelerated gradient descent and projection onto constraints.
"""
import time
import numpy as np

N = 100
_DEADLINE = time.time() + 115  # Safety margin before 120s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
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


def _logsumexp_max(h, beta):
    """
    Smooth approximation of max(C) using log-sum-exp.
    LSE_beta(C) = (1/beta) * log(sum(exp(beta * C_k)))
    """
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    return m + (1.0 / beta) * np.log(np.sum(np.exp(beta * (cs - m))))


def _logsumexp_max_grad(h, beta):
    """
    Gradient of the smooth max approximation.
    Returns (value, gradient).
    """
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    # Softmax weights
    weights = np.exp(beta * (cs - m))
    weights /= np.sum(weights)
    
    value = m + (1.0 / beta) * np.log(np.sum(weights))
    
    # Gradient: sum_k weights_k * dC_k/dh
    # dC_k/dh_j = -(2/n) * (h_{j-k} (if j>=k) + h_{j+k} (if j+k<n))
    n = len(h)
    grad = np.zeros(n)
    factor = (2.0 / n)
    
    for k in range(n):
        w = weights[k]
        if w < 1e-10:
            continue
        if k > 0:
            grad[k:] -= w * factor * h[:n-k]
        if k < n:
            grad[:n-k] -= w * factor * h[k:]
    
    return value, grad


def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h


def _project_onto_box_and_sum(h):
    """
    Project h onto the set {x : 0 <= x_i <= 1, sum(x) = N/2} using
    alternating projections (POCS).
    """
    h = np.array(h, dtype=np.float64, copy=True)
    for _ in range(30):
        # Project onto box
        h = np.clip(h, 0.0, 1.0)
        # Project onto sum constraint
        diff = N / 2.0 - np.sum(h)
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
    return h


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # 1. Analytical Starts
    i = np.arange(N)
    starts = []
    
    # Cosine wave (freq 1)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    starts.append(h_cos)
    
    # Sine wave (freq 1)
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    starts.append(h_sin)
    
    # Square wave (approximated by sign of sine, then smoothed)
    h_sq = 0.5 + 0.5 * np.sign(np.sin(2.0 * np.pi * i / N))
    starts.append(h_sq)
    
    # High frequency cosine (freq 2)
    h_cos2 = 0.5 + 0.5 * np.cos(4.0 * np.pi * i / N)
    starts.append(h_cos2)
    
    # Uniform
    starts.append(np.full(N, 0.5))
    
    # Random starts
    np.random.seed(42)
    for _ in range(3):
        h_rand = np.random.uniform(0, 1, N)
        starts.append(h_rand)
        
    # Optimization parameters
    beta_start = 5.0
    beta_end = 200.0
    
    # Define a helper function to run optimization from a start
    def _optimize_start(h_init, n_iters):
        nonlocal best_h, best_score
        h = _fix_sum(h_init.copy())
        _update_best(h)
        
        v = np.zeros(N)
        t_prev = 1.0
        
        for it in range(n_iters):
            if time.time() - t0 > _DEADLINE - 5:
                return
            
            # Anneal beta: slow start, faster end
            progress = it / n_iters
            # Use a power law for smoother annealing
            beta = beta_start + (beta_end - beta_start) * (progress ** 0.7)
            
            # Compute gradient of smooth max
            _, grad = _logsumexp_max_grad(h, beta)
            
            # Nesterov update
            if it == 0:
                t_curr = 1.0
                y = h
            else:
                t_curr = (1.0 + np.sqrt(1.0 + 8.0 * it)) / 2.0
                y = h + (t_curr - 1.0) / t_curr * v
            
            # Step size: adaptive based on gradient norm
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                step = 0.05 / (grad_norm + 1.0)
            else:
                step = 0.05
            
            # Gradient step in y-space
            h_new = y - step * grad
            h_new = _project_onto_box_and_sum(h_new)
            
            # Update velocity
            v = h_new - h
            h = h_new
            t_prev = t_curr
            
            _update_best(h)
            
    # Run optimization for each start
    # Allocate more iterations to the most promising starts (analytical)
    for idx, h_start in enumerate(starts):
        if time.time() - t0 > _DEADLINE - 10:
            break
            
        if idx < 4: # Analytical starts get more iterations
            iters = 3000
        else:
            iters = 1500
            
        _optimize_start(h_start, iters)
        
    # Final polish: High beta, small steps, many iterations on the best solution
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        h = best_h.copy()
        v = np.zeros(N)
        t_prev = 1.0
        beta_polish = 500.0
        
        for it in range(2000):
            if time.time() - t0 > _DEADLINE - 2:
                break
                
            _, grad = _logsumexp_max_grad(h, beta_polish)
            
            if it == 0:
                t_curr = 1.0
                y = h
            else:
                t_curr = (1.0 + np.sqrt(1.0 + 8.0 * it)) / 2.0
                y = h + (t_curr - 1.0) / t_curr * v
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                # Smaller steps for polish
                step = 0.01 / (grad_norm + 1.0)
            else:
                step = 0.01
            
            h_new = y - step * grad
            h_new = _project_onto_box_and_sum(h_new)
            
            v = h_new - h
            h = h_new
            t_prev = t_curr
            
            _update_best(h)
            
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
