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

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s/150s limit
    
    best_h = np.zeros(N)
    best_ratio = 0.0
    
    # Initialize with uniform distribution
    h = np.ones(N) / N
    r = _compute_ratio(h)
    if r > best_ratio:
        best_ratio = r
        best_h = h.copy()
        
    # Also check delta as baseline
    h_delta = np.zeros(N)
    h_delta[0] = 1.0
    r_delta = _compute_ratio(h_delta)
    if r_delta > best_ratio:
        best_ratio = r_delta
        best_h = h_delta.copy()

    # Strategy: Iterative Multiplicative Updates (Sinkhorn-style for Game)
    # We want to minimize sup(h*h) relative to ||h||_2^2.
    # Let's fix ||h||_1 = 1. We maximize ||h||_2^2 / sup(h*h).
    # Update rule: Reduce h_i where it contributes to the current peak lag.
    
    eta = 0.1  # Learning rate
    max_iters = 2000
    patience = 100
    stale = 0
    
    current_h = h.copy()
    current_ratio = r
    
    for i in range(max_iters):
        if time.time() > deadline - 5.0:
            break
            
        # 1. Compute autoconvolution
        conv = np.convolve(current_h, current_h, mode='full')
        sup_conv = np.max(conv)
        
        if sup_conv < 1e-15:
            break
            
        # 2. Identify the lag(s) that maximize the convolution
        # We can use all lags or just the argmax. Using all lags with weights might be smoother.
        # Let's use the argmax lag for a sharper "attacker" move.
        k_star = np.argmax(conv)
        
        # 3. Compute gradient of the peak convolution w.r.t h
        # (h*h)_k = sum_j h_j h_{j+k}
        # d/dh_i (h*h)_k = h_{i-k} + h_{i+k}
        
        # We need to handle boundaries for the convolution index.
        # np.convolve(h, h, 'full') has length 2N-1.
        # Index k in conv corresponds to sum_{j} h_j h_{j+k-N+1}? 
        # Actually, for 'full' mode:
        # conv[k] = sum_{j} h[j] h[k-j]
        # The index k ranges from 0 to 2N-2.
        # The "lag" in the sense of correlation is usually centered.
        # Let's stick to the gradient of conv[k_star].
        
        # Let's compute the gradient vector grad_h of length N.
        grad_h = np.zeros(N)
        
        # conv[k] = sum_{j=0}^{N-1} h[j] * h[k-j] (where h is 0 outside [0, N-1])
        # d conv[k] / d h[i] = h[k-i] + h[i-k] (if we treat h as symmetric? No, h is not symmetric)
        # Wait, conv is defined as (h*h)[n] = sum_m h[m] h[n-m].
        # d/dh[i] sum_m h[m] h[n-m] 
        # If m=i, term is h[i] h[n-i] -> derivative h[n-i].
        # If n-m=i => m=n-i, term is h[n-i] h[i] -> derivative h[n-i].
        # So d/dh[i] conv[n] = h[n-i] + h[i-n] (if indices valid).
        
        n_star = k_star
        for i_idx in range(N):
            val = 0.0
            # Term 1: h[n_star - i]
            idx1 = n_star - i_idx
            if 0 <= idx1 < N:
                val += current_h[idx1]
            # Term 2: h[i - n_star]
            idx2 = i_idx - n_star
            if 0 <= idx2 < N:
                val += current_h[idx2]
            grad_h[i_idx] = val
            
        # 4. Multiplicative Update
        # h_i <- h_i * exp(-eta * grad_h_i)
        # This reduces h_i where it contributes to the peak.
        
        # To prevent underflow/overflow and maintain scale, we normalize after update.
        # Also, we want to maximize ||h||_2^2. 
        # Reducing mass in "overlapping" areas might shift mass to "isolated" areas, increasing L2.
        
        update = np.exp(-eta * grad_h)
        current_h = current_h * update
        
        # Renormalize to sum=1
        s = np.sum(current_h)
        if s < 1e-15:
            break
        current_h /= s
        
        # 5. Evaluate
        r_new = _compute_ratio(current_h)
        
        if r_new > best_ratio:
            best_ratio = r_new
            best_h = current_h.copy()
            stale = 0
        else:
            stale += 1
            # If we stagnate, try a different eta or restart
            if stale > patience:
                # Try a random restart or different eta
                # Let's try a "spiky" initialization to escape local minima
                # e.g., alternating high/low
                if i % 200 == 0:
                    # Random sparse initialization
                    mask = np.random.rand(N) > 0.8
                    if not np.any(mask):
                        mask[0] = True
                    current_h = mask.astype(float)
                    current_h /= np.sum(current_h)
                    stale = 0
                else:
                    # Decay eta
                    eta *= 0.95
                    if eta < 1e-4:
                        break
                        
    # Final check
    final_ratio = _compute_ratio(best_h)
    if final_ratio > best_ratio:
        best_ratio = final_ratio

    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(_compute_ratio(h))
