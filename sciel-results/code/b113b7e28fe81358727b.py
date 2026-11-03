"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START

def _compute_overlaps(h):
    """
    Compute L_k for all k = 1..N-1
    L_k = sum_{i=k}^{N-1} h_i * (1 - h_{i-k})
    """
    L = np.zeros(N)
    for k in range(1, N):
        L[k] = np.sum(h[k:] * (1 - h[:-k]))
    return L

def _gradient_soft_max(h, tau=5.0):
    """
    Compute gradient of soft max of overlaps w.r.t h
    """
    L = _compute_overlaps(h)
    max_L = np.max(L)
    exp_L = np.exp((L - max_L) * tau)
    sum_exp = np.sum(exp_L)
    weights = exp_L / sum_exp
    
    grad = np.zeros(N)
    for k in range(1, N):
        w = weights[k]
        if w == 0: continue
        
        dLk = np.zeros(N)
        # d/dh_m (sum_{i=k}^{N-1} h_i) = 1 if k <= m < N
        dLk[k:] += 1.0
        # d/dh_m (- sum_{i=k}^{N-1} h_i h_{i-k})
        # Term h_m h_{m-k} contributes -h_{m-k} for m >= k
        dLk[k:] -= h[:N-k]
        # Term h_{m+k} h_m contributes -h_{m+k} for m+k < N
        dLk[:N-k] -= h[k:]
        
        grad += w * dLk
        
    return grad

def _project(h):
    """
    Project h to 0<=h<=1 and sum(h)=N/2
    """
    h = np.clip(h, 0, 1)
    target = N / 2.0
    current_sum = np.sum(h)
    diff = target - current_sum
    
    if abs(diff) < 1e-9:
        return h
        
    for _ in range(20):
        if diff > 0:
            mask = h < 1.0
            if not np.any(mask): break
            add_per = diff / np.sum(mask)
            h[mask] += add_per
        else:
            mask = h > 0.0
            if not np.any(mask): break
            sub_per = -diff / np.sum(mask)
            h[mask] -= sub_per
            
        h = np.clip(h, 0, 1)
        diff = target - np.sum(h)
        if abs(diff) < 1e-9:
            break
            
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    i = np.arange(N)
    # Start with a cosine wave which we hypothesized is good
    h = 0.5 + 0.35 * np.cos(2 * np.pi * i / N)
    h = _project(h)
    
    best_h = h.copy()
    best_score = np.max(_compute_overlaps(best_h))
    
    lr = 0.01
    tau = 5.0
    stagnation_count = 0
    
    while time.time() < deadline:
        grad = _gradient_soft_max(h, tau)
        h_new = h - lr * grad
        h_new = _project(h_new)
        
        current_max = np.max(_compute_overlaps(h_new))
        
        if current_max < best_score - 1e-7:
            best_score = current_max
            best_h = h_new.copy()
            lr *= 1.05
            stagnation_count = 0
        else:
            lr *= 0.9
            stagnation_count += 1
            
            if stagnation_count > 50:
                # Try a small random perturbation
                h_new = h + 0.01 * np.random.randn(N)
                h_new = _project(h_new)
                current_max = np.max(_compute_overlaps(h_new))
                if current_max < best_score:
                    best_score = current_max
                    best_h = h_new.copy()
                    h = h_new
                else:
                    h = best_h.copy()
                lr = 0.01
                stagnation_count = 0
                
        h = h_new
        
    return best_h

# EVOLVE-BLOCK-END
