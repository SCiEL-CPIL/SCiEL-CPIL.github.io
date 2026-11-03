import numpy as np
import time

N = 50

def _compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * sup|h*h|)"""
    h = np.asarray(h, dtype=float)
    if np.all(h < 1e-15):
        return 0.0
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15 or l2_sq < 1e-15:
        return 0.0
    # Autoconvolution
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def _conv_max_at_lag(h, lag):
    """Compute sum_i h[i]*h[i+lag] for a given lag (can be negative)."""
    n = len(h)
    if lag >= 0:
        # i from 0 to n-1-lag
        if lag >= n:
            return 0.0
        return np.sum(h[:n-lag] * h[lag:])
    else:
        # lag < 0, let l = -lag > 0
        # sum_i h[i]*h[i-l] where i ranges from l to n-1
        # = sum_j h[j+l]*h[j] where j ranges from 0 to n-1-l
        l = -lag
        if l >= n:
            return 0.0
        return np.sum(h[l:] * h[:n-l])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_h = np.zeros(N)
    best_ratio = 0.0
    
    def update_best(h):
        nonlocal best_h, best_ratio
        r = _compute_ratio(h)
        if r > best_ratio:
            best_ratio = r
            best_h = h.copy()
        return r
    
    # Initialize with uniform
    h = np.ones(N) / N
    update_best(h)
    
    # Try delta
    h_delta = np.zeros(N)
    h_delta[0] = 1.0
    update_best(h_delta)
    
    # Strategy: Best-response dynamics
    # Fix L1 norm to 1. We want to maximize ||h||_2^2 / max_lag (h*h)_lag
    # where (h*h)_lag = sum_i h[i]*h[i+lag]
    
    # Use a smooth surrogate: soft-max of convolution values
    # Actually, let's use the gradient of the peak with multiplicative updates
    
    eta = 0.05
    max_iters = 3000
    patience = 80
    stale = 0
    current_h = h.copy()
    
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        # Compute full convolution
        conv = np.convolve(current_h, current_h, mode='full')
        
        # conv has length 2N-1
        # conv[k] for k in 0..2N-2
        # The lag-0 autocorrelation is at index N-1
        # lag l (where l>=0) corresponds to index N-1+l
        # lag -l corresponds to index N-1-l
        
        # Find the max convolution value
        sup_conv = np.max(conv)
        if sup_conv < 1e-15:
            break
        
        # Get the index of max
        k_star = np.argmax(conv)
        
        # Convert to lag: lag = k_star - (N-1)
        lag_star = k_star - (N - 1)
        
        # Compute gradient of conv[k_star] w.r.t. h[i]
        # conv[k] = sum_j h[j] * h[k-j] (with h=0 outside [0,N-1])
        # d/dh[i] conv[k] = h[k-i] + h[i-k] (when both indices valid)
        # But actually, since conv is symmetric, d/dh[i] conv[k] = 
        #   (1 if k-i in [0,N-1]) * h[k-i] + (1 if i-k in [0,N-1]) * h[i-k]
        # But wait: conv[k] = sum_{j: 0<=j<N, 0<=k-j<N} h[j]*h[k-j]
        # d/dh[i]: 
        #   If i appears as j: term is h[i]*h[k-i], derivative is h[k-i] (if 0<=k-i<N)
        #   If i appears as k-j, i.e., j=k-i: term is h[k-i]*h[i], derivative is h[k-i] (if 0<=k-i<N)
        #   If i = k-i, i.e., 2i=k, then we only count it once? 
        #   No, in the sum, if j=i and k-j=i, that's the same term h[i]*h[i] = h[i]^2.
        #   The derivative of h[i]^2 w.r.t h[i] is 2*h[i].
        #   But in our formula, we'd get h[k-i] + h[i-k] = h[i] + h[i] = 2*h[i]. Correct.
        
        grad_h = np.zeros(N)
        for i in range(N):
            val = 0.0
            idx1 = k_star - i
            if 0 <= idx1 < N:
                val += current_h[idx1]
            idx2 = i - k_star
            if 0 <= idx2 < N:
                val += current_h[idx2]
            grad_h[i] = val
        
        # Multiplicative update: reduce h where gradient is high
        # h_i <- h_i * exp(-eta * grad_h_i)
        # Then renormalize to sum=1
        
        # For numerical stability, subtract the max gradient
        grad_centered = grad_h - np.max(grad_h)
        update = np.exp(-eta * grad_centered)
        current_h = current_h * update
        
        # Renormalize
        s = np.sum(current_h)
        if s < 1e-15:
            # Reset
            current_h = np.ones(N) / N
            stale = 0
            continue
        current_h /= s
        
        # Ensure nonnegativity
        current_h = np.maximum(current_h, 0)
        current_h /= np.sum(current_h)
        
        # Evaluate
        r_new = _compute_ratio(current_h)
        if r_new > best_ratio:
            best_ratio = r_new
            best_h = current_h.copy()
            stale = 0
        else:
            stale += 1
        
        if stale > patience:
            # Try a perturbation or restart
            if it % 100 == 0:
                # Random sparse initialization
                n_spikes = np.random.randint(3, 15)
                current_h = np.zeros(N)
                indices = np.random.choice(N, size=n_spikes, replace=False)
                current_h[indices] = 1.0
                current_h /= np.sum(current_h)
                stale = 0
            else:
                # Decay eta
                eta *= 0.95
                if eta < 1e-5:
                    break
    
    # Final polish: local search on support
    # Try removing small entries and redistributing
    if time.time() < deadline - 10:
        # Sort by value
        idx_sorted = np.argsort(best_h)[::-1]
        for n_keep in range(2, min(20, N+1)):
            if time.time() > deadline - 5:
                break
            h_trial = np.zeros(N)
            h_trial[idx_sorted[:n_keep]] = best_h[idx_sorted[:n_keep]]
            s = np.sum(h_trial)
            if s > 0:
                h_trial /= s
                r = _compute_ratio(h_trial)
                if r > best_ratio:
                    best_ratio = r
                    best_h = h_trial.copy()
    
    # Final scale optimization: the ratio is scale-invariant? 
    # ||h||_2^2 / (||h||_1 * sup(h*h)) 
    # If h -> c*h: (c^2 * ||h||_2^2) / (c * ||h||_1 * c * sup(h*h)) = same ratio.
    # So scaling doesn't matter. Good.
    
    final_ratio = _compute_ratio(best_h)
    if final_ratio > best_ratio:
        best_ratio = final_ratio
    
    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(_compute_ratio(h))
