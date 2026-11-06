"""Optimized low-discrepancy sign sequence via gradient-based continuous relaxation + discrete local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np

N = 256
DEADLINE = 80.0  # safety margin before 90s kill

def _precompute_aps():
    """Precompute all arithmetic progressions as index arrays."""
    aps = []
    for start in range(N):
        for step in range(1, N):
            # length of AP starting at 'start' with given step
            length = (N - 1 - start) // step + 1
            if length >= 1:
                idx = np.arange(start, N, step)
                aps.append(idx)
    return aps

def _compute_max_abs_ap_sum(signs, ap_indices):
    """Compute max absolute signed sum over all APs."""
    max_val = 0
    for idx in ap_indices:
        s = abs(np.sum(signs[idx]))
        if s > max_val:
            max_val = s
    return max_val

def _soft_loss_and_grad(x, ap_indices, beta):
    """Compute soft-max surrogate loss and gradient.
    x: continuous vector in [-1, 1]
    Returns (loss, grad)
    """
    # Compute all AP sums
    sums = np.array([np.sum(x[idx]) for idx in ap_indices])
    abs_sums = np.abs(sums)
    
    # Soft-max approximation: loss = log(sum(exp(beta * abs_sums))) / beta
    # For numerical stability
    max_abs = np.max(abs_sums)
    shifted = abs_sums - max_abs
    exp_vals = np.exp(beta * shifted)
    loss = (np.log(np.sum(exp_vals)) + beta * max_abs) / beta
    
    # Gradient: d/dx = sum over APs of (beta * exp(beta*|s_i|) / sum_j exp(beta*|s_j|)) * sign(s_i) * e_i
    # where e_i is the indicator of AP i
    weights = beta * exp_vals / np.sum(exp_vals)
    
    # Build gradient
    grad = np.zeros(N)
    for i, idx in enumerate(ap_indices):
        if abs_sums[i] > 1e-10:
            sign_s = np.sign(sums[i])
            grad[idx] += weights[i] * sign_s
    
    return loss, grad

def _local_search(signs, ap_indices, max_time=5.0):
    """Greedy local search: flip single signs if it reduces max AP sum."""
    signs = list(signs)
    start = time.time()
    improved = True
    while improved and (time.time() - start) < max_time:
        improved = False
        best_gain = 0
        best_pos = -1
        current_max = _compute_max_abs_ap_sum(signs, ap_indices)
        
        for pos in range(N):
            # Try flipping
            old = signs[pos]
            signs[pos] = -old
            new_max = _compute_max_abs_ap_sum(signs, ap_indices)
            gain = current_max - new_max
            if gain > best_gain:
                best_gain = gain
                best_pos = pos
            signs[pos] = old
        
        if best_pos >= 0 and best_gain > 0:
            signs[best_pos] = -signs[best_pos]
            improved = True
    
    return signs, _compute_max_abs_ap_sum(signs, ap_indices)

def _pairwise_local_search(signs, ap_indices, max_time=5.0):
    """Local search allowing pairwise flips."""
    signs = list(signs)
    start = time.time()
    improved = True
    while improved and (time.time() - start) < max_time:
        improved = False
        current_max = _compute_max_abs_ap_sum(signs, ap_indices)
        best_gain = 0
        best_pair = None
        
        # Try pairwise flips (sampled to keep it fast)
        rng = np.random.RandomState(42)
        for _ in range(200):
            i, j = rng.choice(N, size=2, replace=False)
            if signs[i] == signs[j]:
                continue
            signs[i] = -signs[i]
            signs[j] = -signs[j]
            new_max = _compute_max_abs_ap_sum(signs, ap_indices)
            gain = current_max - new_max
            if gain > best_gain:
                best_gain = gain
                best_pair = (i, j)
            signs[i] = -signs[i]
            signs[j] = -signs[j]
        
        if best_pair is not None and best_gain > 0:
            i, j = best_pair
            signs[i] = -signs[i]
            signs[j] = -signs[j]
            improved = True
    
    return signs, _compute_max_abs_ap_sum(signs, ap_indices)

def construct():
    t0 = time.time()
    
    # Precompute AP indices
    ap_indices = _precompute_aps()
    
    best_signs = None
    best_score = float('inf')
    
    # Multiple restarts with gradient-based continuous optimization
    rng = np.random.RandomState(171)
    
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() - t0 > DEADLINE - 20:
            break
        
        # Initialize continuous vector with random values in [-1, 1]
        x = rng.uniform(-1, 1, N)
        
        # Gradient descent with adaptive learning rate
        lr = 0.05
        beta = 2.0  # soft-max temperature
        
        for step in range(300):
            if time.time() - t0 > DEADLINE - 15:
                break
            
            loss, grad = _soft_loss_and_grad(x, ap_indices, beta)
            
            # Update with momentum-like behavior
            x = x - lr * grad
            # Clamp to [-1, 1]
            x = np.clip(x, -1, 1)
            
            # Decay learning rate
            lr *= 0.995
        
        # Discretize
        signs = np.where(x >= 0, 1, -1).astype(int).tolist()
        
        # Local search refinement
        signs, score = _local_search(signs, ap_indices, max_time=3.0)
        if score < best_score:
            best_score = score
            best_signs = signs[:]
        
        # Pairwise local search
        if time.time() - t0 < DEADLINE - 10:
            signs, score = _pairwise_local_search(signs, ap_indices, max_time=2.0)
            if score < best_score:
                best_score = score
                best_signs = signs[:]
    
    # Final local search pass if time permits
    if best_signs is not None and time.time() - t0 < DEADLINE - 5:
        best_signs, best_score = _local_search(best_signs, ap_indices, max_time=3.0)
    
    # Ensure we return exactly 256 integers of -1 or +1
    if best_signs is None:
        best_signs = [1] * N
    
    return best_signs
# EVOLVE-BLOCK-END
