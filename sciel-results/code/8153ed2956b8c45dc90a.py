import time
import numpy as np
from scipy.optimize import minimize_scalar

N = 100
_DEADLINE = time.time() + 135  # Safety margin before 150s kill

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

def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective (LogSumExp).
    Returns (value, gradient)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    # Avoid overflow
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for the soft max
    
    val = m + np.log(sum_exp) / beta
    
    grad = np.zeros(n)
    
    # C_k = 1 - (2/n) * r_k
    # r_k = sum_{i=k}^{n-1} h[i]*h[i-k]
    # dC_k/dh_j = -(2/n) * ( h_{j-k} (if j>=k) + h_{j+k} (if j+k<n) )
    # Note: r_0 = sum h_i^2, so dC_0/dh_j = -(2/n) * 2 h_j = -(4/n) h_j
    
    # Vectorized gradient computation
    # For each k, the contribution to grad is -w[k] * (2/n) * (h shifted)
    # Let's construct the gradient by accumulating contributions from each k.
    
    # dC_k/dh_j = -(2/n) * (h_{j-k} + h_{j+k}) (with boundary handling)
    # This is essentially the convolution of h with a symmetric kernel, weighted by w.
    
    # We can compute this using FFT as well, or direct loops if N is small.
    # N=100 is small enough for O(N^2) if vectorized, but let's try to be clever.
    
    # grad[j] = - (2/n) * sum_k w[k] * (h_{j-k} + h_{j+k})
    # This is - (2/n) * (h * w)_circular? No, it's linear correlation.
    
    # Let's use the loop for clarity and correctness, N=100 is fast.
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
    # Binary search for tau such that sum(clamp(y + tau, 0, 1)) = target_sum
    lo = np.min(y) - 1.0
    hi = np.max(y) + 1.0
    
    def g(tau):
        x = np.clip(y + tau, 0.0, 1.0)
        return np.sum(x) - target_sum
    
    # Check bounds
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

def _nesterov_optimize(h0, beta, steps=500, lr=0.01):
    """
    Nesterov Accelerated Gradient Descent optimization.
    """
    h = h0.copy()
    target_sum = N / 2.0
    
    # Nesterov state
    y = h.copy()
    t_k = 1.0
    current_lr = lr
    
    prev_val = _smooth_max_grad(h, beta)[0]
    
    for i in range(steps):
        if time.time() > _DEADLINE - 5:
            break
            
        val, grad = _smooth_max_grad(y, beta)
        
        # Nesterov update
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        theta = (t_k - 1.0) / t_next
        
        # Proposed step
        y_new = y - current_lr * grad
        y_new = y_new + theta * (y_new - h)
        
        # Project
        y_new_proj = _project_sum_box(y_new, target_sum)
        
        # Backtracking line search
        new_val, _ = _smooth_max_grad(y_new_proj, beta)
        
        # If function value increases, reduce step size
        step_lr = current_lr
        while new_val > prev_val + 1e-10 and step_lr > 1e-8:
            step_lr /= 2.0
            y_new = y - step_lr * grad
            y_new = y_new + theta * (y_new - h)
            y_new_proj = _project_sum_box(y_new, target_sum)
            new_val, _ = _smooth_max_grad(y_new_proj, beta)
            
        if new_val < prev_val:
            h = y_new_proj
            y = h.copy() # Reset y to h if we accepted? 
            # Standard NAG: y_{k+1} = x_{k+1} + theta_k (x_{k+1} - x_k)
            # Here x_{k+1} is y_new_proj.
            # So y for next iteration is y_new_proj + theta*(y_new_proj - h_old)
            # But we updated h to y_new_proj.
            # Let's stick to the standard formulation carefully.
            
            # Actually, simpler NAG:
            # 1. y_k = x_k + alpha_{k-1} (x_k - x_{k-1})
            # 2. x_{k+1} = P(y_k - lr * grad(y_k))
            
            # Let's re-implement the loop logic to be safe.
            pass # We will use a simpler momentum approach if NAG gets tricky with projection
            
        else:
            current_lr = max(current_lr * 0.5, 1e-6)
            
        t_k = t_next
        prev_val = new_val
        
    return h

def _pgd_optimize(h0, beta, steps=500, lr=0.01):
    """
    Projected Gradient Descent optimization with momentum.
    """
    h = h0.copy()
    v = np.zeros_like(h)
    target_sum = N / 2.0
    momentum = 0.9
    current_lr = lr
    prev_val = _smooth_max_grad(h, beta)[0]
    
    for i in range(steps):
        if time.time() > _DEADLINE - 5:
            break
            
        val, grad = _smooth_max_grad(h, beta)
        
        # Momentum update
        v = momentum * v - current_lr * grad
        new_h = h + v
        new_h_proj = _project_sum_box(new_h, target_sum)
        
        new_val, _ = _smooth_max_grad(new_h_proj, beta)
        
        # Backtracking
        step_lr = current_lr
        while new_val > prev_val + 1e-10 and step_lr > 1e-8:
            step_lr /= 2.0
            v = momentum * v - step_lr * grad
            new_h = h + v
            new_h_proj = _project_sum_box(new_h, target_sum)
            new_val, _ = _smooth_max_grad(new_h_proj, beta)
            
        if new_val < prev_val:
            h = new_h_proj
            prev_val = new_val
            # Decay momentum slightly to avoid oscillation? Or keep constant?
            # current_lr = min(current_lr * 1.01, 0.1)
        else:
            current_lr = max(current_lr * 0.5, 1e-6)
            v = np.zeros_like(h) # Reset momentum on failure
            
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
    
    # Cosine wave (often good for symmetric problems)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _project_sum_box(h_cos, N / 2.0)
    _update_best(h_cos)
    
    # Sine wave
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    h_sin = _project_sum_box(h_sin, N / 2.0)
    _update_best(h_sin)
    
    # Random starts
    for _ in range(5):
        h_rand = np.random.uniform(0, 1, N)
        h_rand = _project_sum_box(h_rand, N / 2.0)
        _update_best(h_rand)
        
    # 2. Optimization Phases
    # Phase 1: Coarse optimization with lower beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 60:
        h_opt = _pgd_optimize(best_h, beta=10.0, steps=300, lr=0.05)
        _update_best(h_opt)
        
    # Phase 2: Medium beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 30:
        h_opt = _pgd_optimize(best_h, beta=50.0, steps=500, lr=0.01)
        _update_best(h_opt)
        
    # Phase 3: High beta (sharper max approximation)
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        h_opt = _pgd_optimize(best_h, beta=100.0, steps=500, lr=0.005)
        _update_best(h_opt)
        
    # Phase 4: Ultra-high beta polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 2:
        h_opt = _pgd_optimize(best_h, beta=200.0, steps=200, lr=0.001)
        _update_best(h_opt)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    # Final projection to ensure exact sum and bounds
    best_h = _project_sum_box(best_h, N / 2.0)
    
    return best_h
