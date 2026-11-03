"""Optimized program for Erdos minimum overlap task."""
N = 100
import numpy as np
import time

def _compute_overlap(h):
    """Compute the maximum overlap of h with all shifted (1-h).
    
    overlap(k) = sum_i h[i] * (1 - h[(i+k) % N])
    We want to minimize max_k overlap(k)
    """
    one_minus_h = 1.0 - h
    # Compute all cyclic cross-correlations efficiently using FFT
    # overlap(k) = sum_i h[i] * one_minus_h[(i+k) % N]
    # This is the circular cross-correlation of h and one_minus_h
    n = len(h)
    # Use FFT for O(N log N) computation
    F_h = np.fft.rfft(h, n=2*n)
    F_omh = np.fft.rfft(one_minus_h, n=2*n)
    # Circular cross-correlation
    corr = np.fft.irfft(F_h * np.conj(F_omh), n=2*n)[:n]
    # corr[k] = sum_i h[i] * one_minus_h[(i-k) % n] ... need to check sign
    # Actually let's be careful. Let's just do it directly for correctness and speed.
    # For N=100, direct is 100*100 = 10000 ops, which is fine.
    overlaps = np.zeros(n)
    for k in range(n):
        overlaps[k] = np.sum(h * one_minus_h[np.roll(np.arange(n), -k)])
    return np.max(overlaps)

def _compute_overlaps(h):
    """Compute all overlaps for a given h."""
    one_minus_h = 1.0 - h
    n = len(h)
    overlaps = np.zeros(n)
    for k in range(n):
        overlaps[k] = np.sum(h * one_minus_h[np.roll(np.arange(n), -k)])
    return overlaps

def _project_to_simplex_sum(h, target_sum):
    """Project h onto the set {x in [0,1]^n : sum(x) = target_sum} approximately.
    
    We use a simple approach: clip to [0,1], then adjust to meet the sum constraint.
    """
    h = np.clip(h, 0.0, 1.0)
    diff = target_sum - np.sum(h)
    if abs(diff) < 1e-12:
        return h
    # Distribute the difference evenly, but respect bounds
    # Simple iterative approach
    for _ in range(100):
        diff = target_sum - np.sum(h)
        if abs(diff) < 1e-14:
            break
        # Find which elements can be adjusted
        if diff > 0:
            # Need to increase; can only increase elements < 1
            mask = h < 1.0 - 1e-15
            if not np.any(mask):
                break
            h[mask] += diff / np.sum(mask)
        else:
            # Need to decrease; can only decrease elements > 0
            mask = h > 1e-15
            if not np.any(mask):
                break
            h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n = N
    target_sum = n / 2.0
    
    # Initialize with 0.5
    h = np.full(n, 0.5)
    
    # We want to minimize max_k overlap(k)
    # overlap(k) = sum_i h[i] * (1 - h[(i+k)%n])
    #            = sum_i h[i] - sum_i h[i]*h[(i+k)%n]
    #            = n/2 - sum_i h[i]*h[(i+k)%n]   (since sum h = n/2)
    # So minimizing max overlap is equivalent to minimizing max_k (n/2 - autocorr(k))
    # which is equivalent to maximizing min_k autocorr(k)
    # where autocorr(k) = sum_i h[i]*h[(i+k)%n]
    
    # So we want to maximize the minimum cyclic autocorrelation.
    
    # Use gradient-based optimization with a smooth surrogate for min.
    # Let f(h) = min_k autocorr(k). We use soft-min: -log-sum-exp(-autocorr/beta) / beta
    # Or we can use a differentiable approximation.
    
    # Let's use the epigraph formulation:
    # Minimize t subject to: t >= n/2 - autocorr(k) for all k, and sum(h) = n/2, 0<=h<=1
    # This is equivalent to minimizing max_k (n/2 - autocorr(k))
    
    # For a smooth approach, let's use:
    # objective = log(sum(exp(beta * (n/2 - autocorr(k))))) / beta
    # This approximates max_k (n/2 - autocorr(k))
    
    # Gradient of autocorr(k) w.r.t. h[i]:
    # autocorr(k) = sum_j h[j]*h[(j+k)%n]
    # d/dh[i] autocorr(k) = h[(i+k)%n] + h[(i-k)%n]  (for k != 0)
    # For k=0: d/dh[i] autocorr(0) = 2*h[i]
    
    def _autocorr(h):
        """Compute all cyclic autocorrelations."""
        ac = np.zeros(n)
        for k in range(n):
            ac[k] = np.sum(h * h[np.roll(np.arange(n), -k)])
        return ac
    
    def _soft_max_overlap(h, beta):
        """Compute soft approximation of max overlap using log-sum-exp."""
        ac = _autocorr(h)
        overlaps = n/2.0 - ac
        # Use log-sum-exp for numerical stability
        m = np.max(overlaps)
        if m > 50/beta:
            return m  # overflow guard
        log_sum = np.log(np.sum(np.exp(beta * (overlaps - m))))
        return m + log_sum / beta
    
    def _soft_max_overlap_grad(h, beta):
        """Compute gradient of soft max overlap."""
        ac = _autocorr(h)
        overlaps = n/2.0 - ac
        m = np.max(overlaps)
        if m > 50/beta:
            # Fallback to hard max gradient
            idx = np.argmax(overlaps)
            # grad of overlap(k) = -d/dh autocorr(k)
            # d/dh[i] autocorr(k) = h[(i+k)%n] + h[(i-k)%n] for k!=0, 2*h[i] for k=0
            g = np.zeros(n)
            k = idx
            if k == 0:
                g[:] = -2.0 * h
            else:
                g[:] = -(h[np.roll(np.arange(n), k)] + h[np.roll(np.arange(n), -k)])
            return m, g
        
        # Soft max gradient
        exp_vals = np.exp(beta * (overlaps - m))
        weights = exp_vals / np.sum(exp_vals)
        # grad of soft_max = sum_k weights[k] * grad overlap(k)
        # grad overlap(k) = -d/dh autocorr(k)
        g = np.zeros(n)
        for k in range(n):
            if weights[k] < 1e-12:
                continue
            if k == 0:
                g += weights[k] * (-2.0 * h)
            else:
                g += weights[k] * (-(h[np.roll(np.arange(n), k)] + h[np.roll(np.arange(n), -k)]))
        val = m + np.log(np.sum(exp_vals)) / beta
        return val, g
    
    # Main optimization loop
    best_h = h.copy()
    best_val = _compute_overlap(h)
    
    # Progressive beta schedule
    betas = [5, 20, 50, 100, 200, 500, 1000, 2000]
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        lr = 0.01 / (1.0 + beta * 0.01)
        max_iters = 500
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            val, grad = _soft_max_overlap_grad(h, beta)
            
            # Add constraint gradient for sum(h) = n/2
            # We'll handle constraints via projection after update
            
            # Gradient descent step
            h_new = h - lr * grad
            
            # Project to feasible set
            h_new = _project_to_simplex_sum(h_new, target_sum)
            
            # Check improvement
            h_new_clipped = np.clip(h_new, 0.0, 1.0)
            # Re-project to ensure sum is correct
            h_new_clipped = _project_to_simplex_sum(h_new_clipped, target_sum)
            
            raw_val = _compute_overlap(h_new_clipped)
            if raw_val < best_val - 1e-10:
                best_h = h_new_clipped.copy()
                best_val = raw_val
            
            h = h_new_clipped
        
        # Update best
        raw_val = _compute_overlap(h)
        if raw_val < best_val:
            best_h = h.copy()
            best_val = raw_val
    
    # Polish pass with very high beta and tiny LR
    if time.time() < deadline - 5.0:
        beta_polish = 5000
        lr_polish = 0.001
        max_iters_polish = 200
        
        h = best_h.copy()
        for it in range(max_iters_polish):
            if time.time() > deadline - 2.0:
                break
            
            val, grad = _soft_max_overlap_grad(h, beta_polish)
            h_new = h - lr_polish * grad
            h_new = _project_to_simplex_sum(h_new, target_sum)
            
            raw_val = _compute_overlap(h_new)
            if raw_val < best_val - 1e-12:
                best_h = h_new.copy()
                best_val = raw_val
            h = h_new
    
    # Final verification and correction
    result = _project_to_simplex_sum(best_h, target_sum)
    # Ensure exact sum
    diff = target_sum - np.sum(result)
    if abs(diff) > 1e-15:
        # Adjust the largest element
        idx = np.argmax(result) if diff > 0 else np.argmin(result)
        result[idx] += diff
    
    return result
