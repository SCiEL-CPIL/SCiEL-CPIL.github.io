import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Initialize h
    h = np.full(N, 0.5)
    
    # Precompute indices for efficiency?
    # For each k, we need pairs (i, i+k)
    
    def compute_correlations(h):
        C = np.zeros(N-1)
        for k in range(1, N):
            # h[0:N-k] * (1 - h[k:N])
            C[k-1] = np.sum(h[:N-k] * (1 - h[k:]))
        return C

    def compute_gradient(h):
        # Gradient of sum(C_k^2)
        # d/dh_i sum_k C_k^2 = 2 sum_k C_k * dC_k/dh_i
        # C_k = sum_{j=0}^{N-1-k} h_j (1 - h_{j+k})
        # dC_k/dh_i:
        # If i is the first index j: (1 - h_{i+k})
        # If i is the second index j+k: -h_{i-k}
        # So for a fixed i, it contributes to C_k where i < N-k (i is first) or i >= k (i is second).
        
        grad = np.zeros(N)
        C = compute_correlations(h)
        
        # This loop is O(N^2)
        for k in range(1, N):
            ck = C[k-1]
            if ck == 0: continue
            
            # Contribution from h[i] as first term in C_k
            # i ranges from 0 to N-1-k
            # dC_k/dh[i] += (1 - h[i+k])
            if N-1-k >= 0:
                grad[:N-k] += 2 * ck * (1 - h[k:])
            
            # Contribution from h[i] as second term in C_k
            # i ranges from k to N-1
            # dC_k/dh[i] += -h[i-k]
            if k < N:
                grad[k:] += 2 * ck * (-h[:N-k])
        
        return grad

    best_h = h.copy()
    best_max_c = np.inf
    
    # Initial evaluation
    C_init = compute_correlations(h)
    best_max_c = np.max(C_init)
    
    # Gradient Descent
    eta = 0.01
    lr_decay = 0.999
    
    for i in range(5000): # Max iterations
        if time.time() > deadline:
            break
        
        grad = compute_gradient(h)
        
        # Update
        h_new = h - eta * grad
        
        # Project to sum constraint
        # 1. Mean shift
        mean_shift = (N/2.0 - np.sum(h_new)) / N
        h_new += mean_shift
        
        # 2. Clip
        h_new = np.clip(h_new, 0.0, 1.0)
        
        # 3. Fix sum after clipping
        current_sum = np.sum(h_new)
        diff = N/2.0 - current_sum
        
        if diff > 1e-9:
            # Add to elements < 1
            mask = h_new < 1.0
            count = np.sum(mask)
            if count > 0:
                h_new[mask] += diff / count
                h_new = np.clip(h_new, 0.0, 1.0)
        elif diff < -1e-9:
            # Subtract from elements > 0
            mask = h_new > 0.0
            count = np.sum(mask)
            if count > 0:
                h_new[mask] += diff / count
                h_new = np.clip(h_new, 0.0, 1.0)
        
        h = h_new
        
        # Evaluate
        C = compute_correlations(h)
        max_c = np.max(C)
        
        if max_c < best_max_c:
            best_max_c = max_c
            best_h = h.copy()
            
        # Decay learning rate
        eta *= lr_decay
        
        # If eta is too small, restart with random perturbation?
        if eta < 1e-6:
            # Random restart with small noise
            h = best_h + np.random.randn(N) * 0.01
            h = np.clip(h, 0, 1)
            # Re-project
            mean_shift = (N/2.0 - np.sum(h)) / N
            h += mean_shift
            h = np.clip(h, 0.0, 1.0)
            diff = N/2.0 - np.sum(h)
            if diff > 1e-9:
                mask = h < 1.0
                if np.sum(mask) > 0: h[mask] += diff / np.sum(mask)
            elif diff < -1e-9:
                mask = h > 0.0
                if np.sum(mask) > 0: h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            eta = 0.01
            
    return best_h
