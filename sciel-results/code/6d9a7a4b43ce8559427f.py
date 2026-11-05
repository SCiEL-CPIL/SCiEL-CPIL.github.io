import numpy as np
import time
import random

N = 400

def compute_value(h):
    """Compute the task value."""
    h_sum = np.sum(h)
    if h_sum == 0:
        return float('inf')
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    return abs(2 * N * max_conv / (h_sum ** 2))

def subgradient_descent(h0, max_iters=2000, eta0=0.1, eta_decay=0.999):
    """
    Perform subgradient descent to minimize max(autoconv) subject to sum(h)=1.
    """
    h = h0.copy()
    # Normalize to sum 1
    h = h / np.sum(h)
    
    best_h = h.copy()
    best_val = compute_value(best_h)
    
    eta = eta0
    for i in range(max_iters):
        # Compute convolution
        conv = np.convolve(h, h)
        k_star = np.argmax(conv)
        
        # Compute gradient of conv[k_star] w.r.t h
        # d(conv[k])/dh[j] = h[k-j] + h[j-k] (with 0 if index out of bounds)
        g = np.zeros(N)
        # Term 1: h[k_star - j]
        # For j in 0..N-1, index is k_star - j
        # Valid if 0 <= k_star - j < N => k_star - N < j <= k_star
        # We can use slicing or a loop. Slicing is faster.
        
        # Let's construct the gradient vector efficiently.
        # g[j] = h[k_star - j] + h[j - k_star]
        
        # Part 1: h[k_star - j]
        # j ranges 0 to N-1.
        # idx1 = k_star - j
        # We need to add h[idx1] to g[j] if idx1 is in [0, N-1].
        
        # We can create a full reversed h and shift it?
        # h_rev = h[::-1]
        # h[k_star - j] is equivalent to shifting h_rev.
        
        # Simpler: use a temporary array of size 2N-1?
        # Or just use np.roll? No, boundaries are tricky.
        
        # Let's do it with explicit index handling using vectorized operations.
        j = np.arange(N)
        idx1 = k_star - j
        idx2 = j - k_star
        
        mask1 = (idx1 >= 0) & (idx1 < N)
        mask2 = (idx2 >= 0) & (idx2 < N)
        
        g[mask1] += h[idx1[mask1]]
        g[mask2] += h[idx2[mask2]]
        
        # Update
        h = h - eta * g
        
        # Project to sum 1
        h = h - (np.sum(h) - 1) / N
        
        # Update step size
        eta *= eta_decay
        
        # Check value
        val = compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    return best_h, best_val

def construct():
    start_time = time.time()
    time_limit = 140  # Safety margin
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Try a few random starts
    random.seed(42)
    np.random.seed(42)
    
    num_starts = 10
    max_iters_per_start = 5000
    
    for i in range(num_starts):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization
        h0 = np.random.randn(N)
        h0 += 1.0 # Ensure positive mean initially to avoid issues, though negative is allowed
        # Normalize
        h0 = h0 / np.sum(h0)
        
        h_opt, val_opt = subgradient_descent(h0, max_iters=max_iters_per_start, eta0=0.5, eta_decay=0.9995)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
    
    # Final check
    final_val = compute_value(best_h)
    if final_val < best_val:
        best_val = final_val
        
    return best_h
