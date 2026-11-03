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
    deadline = start_time + 140.0  # Safety margin
    
    # Initialize with the known strong static solution from Inspiration 1
    # h = [2/3, 1/3, 0, ...]
    best_h = np.zeros(N)
    best_h[0] = 2.0 / 3.0
    best_h[1] = 1.0 / 3.0
    best_ratio = _compute_ratio(best_h)
    
    # Also check uniform and delta as baselines
    h_uniform = np.ones(N) / N
    r_uniform = _compute_ratio(h_uniform)
    if r_uniform > best_ratio:
        best_ratio = r_uniform
        best_h = h_uniform.copy()
        
    h_delta = np.zeros(N)
    h_delta[0] = 1.0
    r_delta = _compute_ratio(h_delta)
    if r_delta > best_ratio:
        best_ratio = r_delta
        best_h = h_delta.copy()

    # Strategy: Iterative Multiplicative Updates (Sinkhorn-style for Game)
    # Starting from the best known seed
    
    eta = 0.1  # Learning rate
    max_iters = 5000
    patience = 100
    stale = 0
    
    current_h = best_h.copy()
    current_ratio = best_ratio
    
    for i in range(max_iters):
        if time.time() > deadline - 5.0:
            break
            
        # 1. Compute autoconvolution
        conv = np.convolve(current_h, current_h, mode='full')
        sup_conv = np.max(conv)
        
        if sup_conv < 1e-15:
            break
            
        # 2. Identify the lag(s) that maximize the convolution
        k_star = np.argmax(conv)
        
        # 3. Compute gradient of the peak convolution w.r.t h
        grad_h = np.zeros(N)
        n_star = k_star
        for idx in range(N):
            val = 0.0
            idx1 = n_star - idx
            if 0 <= idx1 < N:
                val += current_h[idx1]
            idx2 = idx - n_star
            if 0 <= idx2 < N:
                val += current_h[idx2]
            grad_h[idx] = val
            
        # 4. Multiplicative Update
        # Reduce h_i where it contributes to the peak.
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
                # Try a "spiky" initialization to escape local minima
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
                        
    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(_compute_ratio(h))
