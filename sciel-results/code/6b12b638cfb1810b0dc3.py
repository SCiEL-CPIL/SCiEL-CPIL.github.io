"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _overlap(h):
    """Compute overlap C_k = sum_i h[i]*(1-h[(i+k)%N]) for all k."""
    # C_k = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
    # sum_i h[i] = N/2 (if normalized)
    # So C_k = N/2 - corr_k where corr_k = sum_i h[i]*h[(i+k)%N]
    # We want to minimize max_k C_k = N/2 - min_k corr_k
    # So we want to maximize min_k corr_k
    
    # Compute all circular correlations
    # corr_k = sum_i h[i] * h[(i+k)%N]
    # This is the circular cross-correlation of h with itself
    # Use FFT for efficiency
    n = len(h)
    h_pad = np.zeros(2*n)
    h_pad[:n] = h
    # Circular correlation: corr_k = sum_i h[i] * h[(i+k)%n]
    # This is equivalent to: for each k, sum over i of h[i]*h[i+k mod n]
    # Using FFT: corr = ifft(fft(h) * conj(fft(h))) gives autocorrelation
    # But we need circular correlation of h with h shifted
    
    # Actually, let's just do it directly for clarity, N=100 is small
    # But for speed in optimization, FFT is better
    
    # Let's use the direct method first, it's O(N^2) = 10000 ops, very fast
    corr = np.zeros(n)
    for k in range(n):
        corr[k] = np.dot(h, np.roll(h, k))
    
    # C_k = sum(h) - corr_k
    # If sum(h) = n/2, then C_k = n/2 - corr_k
    # We want to minimize max(C_k) = n/2 - min(corr_k)
    # So minimize max(C_k) <=> maximize min(corr_k)
    
    return n/2.0 - corr  # This gives C_k for each k

def _max_overlap(h):
    """Return the maximum overlap value."""
    c = _overlap(h)
    return np.max(c)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial guess: uniform 0.5
    h = np.full(N, 0.5)
    
    # Normalize to ensure sum = N/2
    h = h / np.sum(h) * (N/2.0)
    h = np.clip(h, 0, 1)
    
    # Coordinate descent with line search on pairs
    # Objective: minimize max_k C_k where C_k = sum_i h[i]*(1-h[(i+k)%N])
    
    # Use a smooth surrogate: soft-max via log-sum-exp
    # f(h) = log(sum_k exp(beta * C_k)) / beta
    # This approximates max(C_k) as beta -> inf
    
    best_h = h.copy()
    best_val = _max_overlap(h)
    
    def soft_max_obj(h, beta):
        c = _overlap(h)
        # Shift for numerical stability
        c_shifted = c - np.max(c)
        log_sum = np.log(np.sum(np.exp(beta * c_shifted)))
        return (np.max(c) + log_sum) / beta
    
    def soft_max_grad(h, beta):
        c = _overlap(h)
        c_shifted = c - np.max(c)
        exp_vals = np.exp(beta * c_shifted)
        probs = exp_vals / np.sum(exp_vals)
        # Gradient of soft-max objective w.r.t. h
        # d/dh_i [log sum_k exp(beta * C_k)] = sum_k probs_k * beta * dC_k/dh_i
        # C_k = sum_j h[j] - sum_j h[j]*h[(j+k)%N]
        # dC_k/dh_i = 1 - h[(i-k)%N] - h[(i+k)%N] ... wait let me re-derive
        # C_k = sum_j h[j]*(1 - h[(j+k)%N])
        # dC_k/dh_i: terms where i appears
        # Term 1: h[i]*(1-h[(i+k)%N]) -> derivative w.r.t h[i] is (1-h[(i+k)%N])
        # Term 2: for j such that (j+k)%N = i, i.e., j = (i-k)%N
        #   h[(i-k)%N]*(1-h[i]) -> derivative w.r.t h[i] is -h[(i-k)%N]
        # So dC_k/dh_i = (1 - h[(i+k)%N]) - h[(i-k)%N]
        
        # grad_obj_i = sum_k probs_k * beta * [(1 - h[(i+k)%N]) - h[(i-k)%N]]
        grad = np.zeros(N)
        for i in range(N):
            s = 0.0
            for k in range(N):
                dk_di = (1 - h[(i+k)%N]) - h[(i-k)%N]
                s += probs[k] * beta * dk_di
            grad[i] = s
        return grad
    
    # Coordinate descent with 2D line searches
    # For each pair (i, j), do a 2D line search to minimize the soft-max objective
    # Subject to 0 <= h <= 1 and sum(h) = N/2
    
    # Actually, let's use a simpler approach: projected gradient descent
    # with adaptive beta
    
    beta = 10.0
    lr = 0.1
    iters_per_beta = 200
    
    for beta_stage in range(1, 20):
        beta = beta_stage * 5.0
        for it in range(iters_per_beta):
            if time.time() > deadline - 5.0:
                break
            
            # Compute gradient
            grad = soft_max_grad(h, beta)
            
            # Projected gradient step
            h_new = h - lr * grad
            h_new = np.clip(h_new, 0, 1)
            
            # Normalize to maintain sum = N/2
            s = np.sum(h_new)
            if s > 0:
                h_new = h_new / s * (N/2.0)
                h_new = np.clip(h_new, 0, 1)
            
            # Check if it's better
            val = soft_max_obj(h_new, beta)
            val_old = soft_max_obj(h, beta)
            
            if val < val_old:
                h = h_new
            else:
                # Try smaller step
                lr = lr * 0.5
                h_new = h - lr * grad
                h_new = np.clip(h_new, 0, 1)
                s = np.sum(h_new)
                if s > 0:
                    h_new = h_new / s * (N/2.0)
                    h_new = np.clip(h_new, 0, 1)
                val = soft_max_obj(h_new, beta)
                if val < val_old:
                    h = h_new
            
            # Track best
            raw_val = _max_overlap(h)
            if raw_val < best_val:
                best_val = raw_val
                best_h = h.copy()
        
        if time.time() > deadline - 5.0:
            break
        
        # Adjust learning rate
        lr = max(lr * 1.1, 0.01)
    
    # Final polish: coordinate descent on individual coordinates
    if time.time() < deadline - 10.0:
        for _ in range(500):
            if time.time() > deadline - 10.0:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 10.0:
                    break
                # Try moving h[i] slightly
                for delta in [-0.01, -0.005, 0.005, 0.01]:
                    h_test = h.copy()
                    h_test[i] += delta
                    if h_test[i] < 0 or h_test[i] > 1:
                        continue
                    # Renormalize
                    s = np.sum(h_test)
                    if s <= 0:
                        continue
                    h_test = h_test / s * (N/2.0)
                    h_test = np.clip(h_test, 0, 1)
                    val = _max_overlap(h_test)
                    if val < best_val - 1e-12:
                        h = h_test
                        best_val = val
                        best_h = h.copy()
                        improved = True
                        break
    
    # Ensure exact normalization
    best_h = best_h / np.sum(best_h) * (N/2.0)
    best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
