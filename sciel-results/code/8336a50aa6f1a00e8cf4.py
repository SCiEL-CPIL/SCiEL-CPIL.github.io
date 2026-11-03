"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _overlaps(h):
    """Compute overlap of h with shifted (1-h) for all circular shifts."""
    N = len(h)
    overlaps = np.zeros(N)
    one_minus_h = 1.0 - h
    for k in range(N):
        shifted = np.roll(one_minus_h, k)
        overlaps[k] = np.sum(h * shifted)
    return overlaps

def _soft_max(overlaps, temp=1.0):
    """Soft-max approximation of max."""
    m = np.max(overlaps)
    return m + temp * np.log(np.sum(np.exp((overlaps - m) / temp)))

def construct():
    start_time = time.time()
    deadline = 150.0  # safety margin
    
    # Start from baseline
    h = np.full(N, 0.5)
    
    # Try to optimize using gradient descent on soft-max of overlaps
    # Use logit parameterization: h_i = sigmoid(x_i)
    x = np.zeros(N)  # logit, h = sigmoid(x)
    
    lr = 0.1
    temp = 1.0
    best_h = h.copy()
    best_score = _soft_max(_overlaps(h), temp)
    
    for it in range(5000):
        if time.time() > deadline:
            break
        
        h = 1.0 / (1.0 + np.exp(-x))
        overlaps = _overlaps(h)
        current_score = _soft_max(overlaps, temp)
        
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
        
        # Gradient of soft-max w.r.t. overlaps
        exp_overlaps = np.exp((overlaps - np.max(overlaps)) / temp)
        weights = exp_overlaps / np.sum(exp_overlaps)
        
        # Gradient of overlap[k] w.r.t. h_i: (1-h_{i+k}) for shift k
        # So grad_h_i = sum_k weights[k] * (1 - h_{i+k})
        grad_h = np.zeros(N)
        for k in range(N):
            shifted_omh = np.roll(1.0 - h, k)
            grad_h += weights[k] * shifted_omh
        
        # Gradient w.r.t. x: d h_i / d x_i = h_i * (1 - h_i)
        grad_x = grad_h * h * (1.0 - h)
        
        # Enforce sum constraint: project or use Lagrange
        # Simple: subtract mean of grad_x to keep sum roughly constant
        grad_x -= np.mean(grad_x)
        
        x -= lr * grad_x
        
        # Anneal temperature
        if it % 500 == 0:
            temp *= 0.5
    
    # Final polish: use best_h found
    # Clip to [0,1] and adjust to ensure sum = N/2 exactly
    h_final = np.clip(best_h, 0.0, 1.0)
    # Adjust to make sum exactly N/2
    current_sum = np.sum(h_final)
    if abs(current_sum - N/2) > 1e-10:
        h_final += (N/2 - current_sum) / N
        h_final = np.clip(h_final, 0.0, 1.0)
        # Re-adjust if clipping changed sum
        diff = N/2 - np.sum(h_final)
        if abs(diff) > 1e-10:
            h_final += diff / N
    
    return h_final
# EVOLVE-BLOCK-END
