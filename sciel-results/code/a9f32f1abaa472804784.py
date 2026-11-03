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
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def _multiplicative_optimize(h0, deadline, max_iters=3000):
    """
    Iterative multiplicative updates to reduce peak autoconvolution while maintaining L2 norm.
    h_i <- h_i * exp(-eta * grad_sup_i)
    Where grad_sup_i is the gradient of the peak convolution w.r.t h_i.
    """
    h = h0.copy()
    l1 = np.sum(h)
    if l1 < 1e-15:
        return h, 0.0
    h = h / l1  # Normalize to L1=1
    
    best_h = h.copy()
    best_ratio = _compute_ratio(h)
    
    eta = 0.05
    stale = 0
    patience = 100
    
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
            
        conv = np.convolve(h, h, mode='full')
        sup_conv = np.max(conv)
        
        if sup_conv < 1e-15:
            break
            
        # Find the lag(s) that maximize convolution
        # Use a soft-max weighting over the top few lags to smooth the gradient
        # This helps escape local minima caused by discrete argmax jumps
        k = np.argmax(conv)
        
        # Compute gradient of conv[k] w.r.t h
        # conv[k] = sum_j h[j] h[k-j]
        # d conv[k] / d h[i] = h[k-i] + h[i-k] (if indices valid)
        grad = np.zeros(N)
        for i in range(N):
            idx1 = k - i
            idx2 = i - k
            if 0 <= idx1 < N:
                grad[i] += h[idx1]
            if 0 <= idx2 < N:
                grad[i] += h[idx2]
        
        # Multiplicative update
        # We want to reduce h_i where it contributes to the peak.
        # To preserve L2 norm somewhat, we can scale the update.
        # However, simple multiplicative descent on the peak is a good start.
        # h_i <- h_i * exp(-eta * grad_i)
        
        # To prevent h from becoming zero, we can add a small floor or use a different update.
        # Let's stick to pure multiplicative for now.
        
        update = np.exp(-eta * grad)
        h_new = h * update
        
        # Renormalize to L1=1
        s = np.sum(h_new)
        if s < 1e-15:
            break
        h_new = h_new / s
        
        r_new = _compute_ratio(h_new)
        
        if r_new > best_ratio:
            best_ratio = r_new
            best_h = h_new.copy()
            stale = 0
            # If we improve, we can slightly increase eta or keep it
            # eta = min(eta * 1.01, 0.5) # Optional
        else:
            stale += 1
            h = h_new # Keep the new h even if ratio didn't improve, to continue moving
            if stale > patience:
                # Restart or perturb
                # Perturb h
                h = h * (1 + 0.1 * np.random.randn(N))
                h[h < 0] = 0
                s = np.sum(h)
                if s > 1e-15:
                    h = h / s
                else:
                    h = h0.copy() / np.sum(h0)
                stale = 0
                eta *= 0.8
                if eta < 1e-5:
                    break
    
    return best_h, best_ratio

def _gradient_optimize(h0, deadline, max_iters=2000):
    """
    Gradient ascent on log(ratio) with smoothing.
    """
    h = h0.copy()
    l1 = np.sum(h)
    if l1 < 1e-15:
        return h, 0.0
    h = h / l1
    
    best_h = h.copy()
    best_ratio = _compute_ratio(h)
    
    beta = 1.0
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
            
        conv = np.convolve(h, h, mode='full')
        m = np.max(conv)
        if m < 1e-15:
            break
            
        # Softmax weights for the convolution peaks
        # We want to minimize the "effective" sup norm.
        # Let's use a weighted average of the top peaks.
        # Or just use the max peak for gradient.
        
        # Gradient of log(l2_sq) - log(l1) - log(sup_conv)
        # Since l1 is fixed to 1, we maximize log(l2_sq) - log(sup_conv)
        
        l2_sq = np.sum(h**2)
        
        # Gradient of l2_sq: 2h
        # Gradient of sup_conv: use the gradient of the argmax conv
        k = np.argmax(conv)
        grad_conv = np.zeros(N)
        for i in range(N):
            idx1 = k - i
            idx2 = i - k
            if 0 <= idx1 < N:
                grad_conv[i] += h[idx1]
            if 0 <= idx2 < N:
                grad_conv[i] += h[idx2]
        
        # Grad of objective:
        # d/dh [ log(l2_sq) - log(conv_k) ]
        # = 2h/l2_sq - grad_conv / conv_k
        
        grad_obj = 2.0 * h / l2_sq - grad_conv / conv[k]
        
        # Project to non-negative and normalize
        # Use a small step
        lr = 0.01 / (1.0 + 0.01 * it)
        h_new = h + lr * grad_obj
        h_new[h_new < 0] = 0
        s = np.sum(h_new)
        if s < 1e-15:
            break
        h_new = h_new / s
        
        r_new = _compute_ratio(h_new)
        if r_new > best_ratio:
            best_ratio = r_new
            best_h = h_new.copy()
        h = h_new
        
    return best_h, best_ratio

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_h = np.zeros(N)
    best_ratio = 0.0
    
    # 1. Initialize with good candidates
    # Uniform
    h_uni = np.ones(N) / N
    r = _compute_ratio(h_uni)
    if r > best_ratio:
        best_ratio = r
        best_h = h_uni.copy()
        
    # Sidon-like (sparse)
    # The seed program used [0, 1, 4, 10, 18, 23, 30]
    sidon_S = [0, 1, 4, 10, 18, 23, 30]
    h_sidon = np.zeros(N)
    for idx in sidon_S:
        h_sidon[idx] = 1.0 / len(sidon_S)
    r = _compute_ratio(h_sidon)
    if r > best_ratio:
        best_ratio = r
        best_h = h_sidon.copy()
        
    # Random sparse
    np.random.seed(42)
    for _ in range(10):
        k = np.random.randint(3, 15)
        idxs = np.random.choice(N, k, replace=False)
        h_r = np.zeros(N)
        h_r[idxs] = 1.0 / k
        r = _compute_ratio(h_r)
        if r > best_ratio:
            best_ratio = r
            best_h = h_r.copy()
            
    # 2. Optimize from multiple starts using both methods
    starts = [h_uni.copy(), h_sidon.copy(), best_h.copy()]
    
    # Add some random perturbations of best
    for _ in range(5):
        h_p = best_h * (1 + 0.2 * np.random.randn(N))
        h_p[h_p < 0] = 0
        s = np.sum(h_p)
        if s > 1e-15:
            h_p = h_p / s
            starts.append(h_p)
            
    for h_start in starts:
        if time.time() > deadline - 15.0:
            break
            
        # Try multiplicative
        h_opt1, r_opt1 = _multiplicative_optimize(h_start, deadline)
        if r_opt1 > best_ratio:
            best_ratio = r_opt1
            best_h = h_opt1.copy()
            
        # Try gradient on the best found so far
        if time.time() > deadline - 10.0:
            break
        h_opt2, r_opt2 = _gradient_optimize(best_h, deadline)
        if r_opt2 > best_ratio:
            best_ratio = r_opt2
            best_h = h_opt2.copy()
            
    # Final polish with multiplicative on best
    if time.time() < deadline - 5.0:
        h_final, r_final = _multiplicative_optimize(best_h, deadline)
        if r_final > best_ratio:
            best_ratio = r_final
            best_h = h_final.copy()

    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(_compute_ratio(h))
