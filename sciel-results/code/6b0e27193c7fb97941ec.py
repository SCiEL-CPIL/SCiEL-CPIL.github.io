"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_overlap_max(h):
    """Compute the maximum overlap of h with shifted 1-h.
    
    The overlap at shift k is sum_i h[i] * (1 - h[(i+k) % N])
    We want to minimize the maximum over all shifts k.
    
    Note: overlap at shift k = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
    Since sum_i h[i] = N/2, overlap = N/2 - autocorr(k)
    where autocorr(k) = sum_i h[i]*h[(i+k)%N]
    
    So minimizing max overlap = minimizing max_k (N/2 - autocorr(k))
    = N/2 - min_k autocorr(k)
    
    Wait, let me reconsider. The overlap of h with shifted (1-h) at shift k:
    overlap(k) = sum_i h[i] * (1 - h[(i+k) % N])
               = sum_i h[i] - sum_i h[i] * h[(i+k) % N]
               = N/2 - autocorr(k)
    
    We want to minimize max_k overlap(k) = max_k (N/2 - autocorr(k))
    = N/2 - min_k autocorr(k)
    
    So we want to maximize min_k autocorr(k).
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    # Compute autocorrelation for all shifts
    autocorr = np.zeros(n)
    for k in range(n):
        autocorr[k] = np.dot(h, np.roll(h, k))
    
    overlap = n / 2.0 - autocorr
    return np.max(overlap)

def _smooth_max(x, tau=0.01):
    """Smooth maximum using log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x)
    if m < -100:
        return m
    return m + tau * np.log(np.sum(np.exp((x - m) / tau)))

def construct():
    start_time = time.time()
    deadline = 130.0  # Leave 20s margin
    
    # Initial guess: constant 0.5 gives overlap = N/2 - N/4 = N/4 = 25
    # We want to do better.
    
    # Strategy: Use gradient descent on the step heights h, 
    # with the constraint that sum(h) = N/2 and 0 <= h <= 1.
    
    # Parameterize h using a smooth function to maintain constraints.
    # h[i] = sigmoid(z[i]) * 0.5 + 0.25 ... no, let's use a different approach.
    
    # Actually, let's directly optimize h with projected gradient descent.
    
    h = np.full(N, 0.5)
    
    # Try a few different initializations
    best_h = h.copy()
    best_score = _compute_overlap_max(h)
    
    # Add some structure: try sinusoidal patterns
    for freq in [1, 2, 3, 4, 5]:
        t = np.arange(N) / N * 2 * np.pi
        h_test = 0.5 + 0.3 * np.sin(freq * t)
        h_test = np.clip(h_test, 0, 1)
        # Normalize to sum = N/2
        h_test = h_test / np.sum(h_test) * (N / 2.0)
        h_test = np.clip(h_test, 0, 1)
        # Re-normalize after clipping
        h_test = h_test / np.sum(h_test) * (N / 2.0)
        
        score = _compute_overlap_max(h_test)
        if score < best_score:
            best_score = score
            best_h = h_test.copy()
    
    # Gradient descent with smooth surrogate
    h = best_h.copy()
    lr = 0.01
    tau = 0.1  # temperature for smooth max
    
    for step in range(5000):
        if time.time() > deadline:
            break
        
        # Compute overlaps
        autocorr = np.zeros(N)
        for k in range(N):
            autocorr[k] = np.dot(h, np.roll(h, k))
        overlap = N / 2.0 - autocorr
        
        # Smooth max gradient
        m = np.max(overlap)
        weights = np.exp((overlap - m) / tau)
        weights /= np.sum(weights)
        
        # Gradient of smooth max w.r.t. overlap[k] is weights[k]
        # overlap[k] = N/2 - sum_i h[i]*h[(i+k)%N]
        # d(overlap[k])/d(h[j]) = -h[(j+k)%N] - h[(j-k)%N]
        
        grad = np.zeros(N)
        for k in range(N):
            if weights[k] > 1e-10:
                grad += -weights[k] * (np.roll(h, -k) + np.roll(h, k))
        
        # Project gradient to maintain sum constraint
        grad -= np.mean(grad)
        
        h_new = h - lr * grad
        h_new = np.clip(h_new, 0, 1)
        h_new = h_new / np.sum(h_new) * (N / 2.0)
        h_new = np.clip(h_new, 0, 1)
        
        score = _compute_overlap_max(h_new)
        if score < best_score:
            best_score = score
            best_h = h_new.copy()
            h = h_new.copy()
            # Reduce learning rate
            lr *= 0.99
        else:
            lr *= 0.95
            if lr < 1e-6:
                break
    
    return best_h
# EVOLVE-BLOCK-END
