"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def compute_overlap(h, k):
    """Compute overlap at shift k (circular)."""
    h_shifted = np.roll(h, -k)
    return np.mean(np.minimum(h, 1.0 - h_shifted))

def compute_max_overlap(h):
    """Compute max overlap over all shifts."""
    max_ov = 0.0
    for k in range(N):
        ov = compute_overlap(h, k)
        if ov > max_ov:
            max_ov = ov
    return max_ov

def logsumexp(x, beta):
    """Stable log-sum-exp."""
    max_x = np.max(x)
    if max_x == -np.inf:
        return -np.inf
    return max_x + np.log(np.sum(np.exp(x - max_x))) / beta * beta  # This is wrong
    # Actually: (1/beta) * log(sum(exp(beta * x)))
    # = max_x + (1/beta) * log(sum(exp(beta*(x - max_x))))

def logsumexp_correct(x, beta):
    max_x = np.max(x)
    return max_x + (1.0 / beta) * np.log(np.sum(np.exp(beta * (x - max_x))))

def compute_smoothed_max_overlap(h, beta):
    """Compute smoothed max overlap using log-sum-exp."""
    overlaps = np.array([compute_overlap(h, k) for k in range(N)])
    return logsumexp_correct(overlaps, beta)

def project(h, target_sum):
    """Project h onto the constraint sum(h) = target_sum, h in [0,1]."""
    # Simple projection: scale and clip, then adjust
    for _ in range(10):
        h_clipped = np.clip(h, 0.0, 1.0)
        diff = target_sum - np.sum(h_clipped)
        if abs(diff) < 1e-10:
            break
        # Distribute the difference evenly
        h_clipped = h_clipped + diff / N
        h_clipped = np.clip(h_clipped, 0.0, 1.0)
    return h_clipped

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    target_sum = N / 2.0
    
    # Initialize with 0.5
    h = np.full(N, 0.5)
    
    # Try multiple initializations
    best_h = h.copy()
    best_score = compute_max_overlap(h)
    
    # Phase 1: Coarse optimization with smooth log-sum-exp
    betas = [10, 50, 100, 200, 500]
    
    for beta in betas:
        if time.time() > deadline - 20:
            break
        
        # Gradient descent with projection
        lr = 0.01
        for step in range(200):
            if time.time() > deadline - 10:
                break
            
            # Compute overlaps for all shifts
            overlaps = np.array([compute_overlap(h, k) for k in range(N)])
            
            # Compute gradient of log-sum-exp
            exp_vals = np.exp(beta * (overlaps - np.max(overlaps)))
            weights = exp_vals / np.sum(exp_vals)
            
            # Gradient of each overlap w.r.t. h
            # overlap(k) = mean(min(h[i], 1-h[(i+k)%N]))
            # d/dh[i] min(h[i], 1-h[(i+k)%N]):
            # If h[i] < 1-h[(i+k)%N], derivative is 1/N
            # If h[i] > 1-h[(i+k)%N], derivative is 0
            # Also h[i] appears in overlap(k') where i = (j+k')%N, i.e., k' = (i-j)%N
            # For those, h[i] is in 1-h[(j+k')%N] = 1-h[i], so derivative is -1/N if 1-h[i] < h[j]
            
            grad = np.zeros(N)
            for k in range(N):
                h_shifted = np.roll(h, -k)
                # Where h[i] <= 1-h_shifted[i], d(overlap)/d(h[i]) = 1/N
                # Where h[i] > 1-h_shifted[i], d(overlap)/d(h[i]) = 0
                mask1 = h <= (1.0 - h_shifted)
                grad += weights[k] * mask1 / N
                
                # h[i] also appears as 1-h[(j+k)%N] for j such that (j+k)%N = i, i.e., j = (i-k)%N
                # In that case, overlap(k) has term min(h[j], 1-h[i])
                # d/dh[i] min(h[j], 1-h[i]) = -1/N if 1-h[i] < h[j], i.e., h[i] > 1-h[j]
                # h[j] = h[(i-k)%N]
                h_prev = np.roll(h, k)  # h[(i-k)%N] for each i
                mask2 = (1.0 - h) < h_prev  # h[i] > 1-h[j]
                grad += weights[k] * (-mask2 / N)
            
            # Update
            h = h - lr * grad
            h = project(h, target_sum)
        
        # Evaluate
        score = compute_max_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Phase 2: Fine polishing with sharp log-sum-exp
    if time.time() < deadline - 5:
        beta = 500
        lr = 0.001
        for step in range(500):
            if time.time() > deadline - 2:
                break
            
            overlaps = np.array([compute_overlap(h, k) for k in range(N)])
            exp_vals = np.exp(beta * (overlaps - np.max(overlaps)))
            weights = exp_vals / np.sum(exp_vals)
            
            grad = np.zeros(N)
            for k in range(N):
                h_shifted = np.roll(h, -k)
                mask1 = h <= (1.0 - h_shifted)
                grad += weights[k] * mask1 / N
                
                h_prev = np.roll(h, k)
                mask2 = (1.0 - h) < h_prev
                grad += weights[k] * (-mask2 / N)
            
            h = h - lr * grad
            h = project(h, target_sum)
        
        score = compute_max_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    return best_h
# EVOLVE-BLOCK-END

