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

def _soft_sup_conv(h, beta):
    """
    Compute the soft-max (log-sum-exp) of the autoconvolution.
    This is a smooth surrogate for max(conv).
    """
    conv = np.convolve(h, h, mode='full')
    # Shift to center for numerical stability if needed, but LSE is shift invariant in exp
    # We want S = sum(exp(beta * conv)). 
    # To avoid overflow, we use the log-sum-exp trick.
    # However, we need the gradient of log(S) or S itself?
    # Objective: Maximize L2^2 / SupConv.
    # If we use SoftSup = log(sum(exp(beta*conv))/N_conv), it's a smooth upper bound on max.
    # Actually, let's just compute the gradient of the soft-max value directly.
    # Let M = max(conv). Let Z = sum(exp(beta*(conv - M))).
    # SoftMax = M + (1/beta) * log(Z).
    # This is smooth and approximates max as beta -> inf.
    
    M = np.max(conv)
    # Subtract M to prevent overflow in exp
    exp_terms = np.exp(beta * (conv - M))
    Z = np.sum(exp_terms)
    # If Z is 0 (shouldn't happen if h is non-zero), fallback to M
    if Z < 1e-15:
        return M, conv, M
    
    soft_val = M + (1.0 / beta) * np.log(Z)
    return soft_val, conv, M

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = np.zeros(N)
    best_ratio = 0.0
    
    # Initialize with uniform distribution
    h = np.ones(N) / N
    
    # Update the best if uniform is good
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

    # Strategy: Gradient Ascent on log(h) with Soft-Max Convolution
    # We maximize J(h) = ||h||_2^2 / S(h) subject to ||h||_1 = 1, h >= 0.
    # Let u = log(h). h = exp(u) / sum(exp(u)).
    
    eta = 0.05  # Learning rate for log-h
    max_iters = 8000
    patience = 150
    stale = 0
    beta = 5.0  # Temperature for soft-max (start lower for smoothness, increase later)
    
    current_h = h.copy()
    current_ratio = r
    log_h = np.log(current_h + 1e-30)
    
    for i in range(max_iters):
        if time.time() > deadline - 10.0:
            break
            
        # 1. Compute Soft-Max Convolution and its gradient
        # S = SoftMax(beta, conv)
        # dS/du = ?
        # It's easier to compute gradient with respect to h, then chain rule to u.
        # Let's stick to h-space for gradient computation but update in log-space?
        # Or just use multiplicative updates in h-space but guided by soft-max.
        
        # Let's compute gradient of Soft-Sup w.r.t h.
        # SoftSup = M + (1/beta) log(sum_j exp(beta(conv_j - M)))
        # d SoftSup / d h_k = (1/beta) * sum_j [ exp(beta(conv_j - M)) / Z * d(conv_j)/d h_k ]
        # Note: dM/dh_k is non-smooth, but in the LSE formulation, if we treat M as fixed 
        # during the derivative of the log-sum part (common approximation), or just use 
        # the full derivative.
        # Actually, the gradient of log(sum(exp(x_j))) is sum(exp(x_j)/sum(exp(x_j)) * dx_j).
        # So d SoftSup / d h_k = sum_j [ w_j * d(conv_j)/d h_k ]
        # where w_j = exp(beta(conv_j - M)) / Z.
        # These weights w_j sum to 1 and are proportional to the "importance" of each lag.
        
        conv = np.convolve(current_h, current_h, mode='full')
        M = np.max(conv)
        exp_terms = np.exp(beta * (conv - M))
        Z = np.sum(exp_terms)
        if Z < 1e-15:
            break
        weights = exp_terms / Z
        
        # Gradient of weighted sum of convolutions
        # grad_h[k] = sum_j weights[j] * d(conv_j)/d h_k
        # d(conv_j)/d h_k = h[j-k] + h[k-j] (with boundary checks)
        
        grad_h = np.zeros(N)
        # We can vectorize this slightly or use loops. N=50 is small.
        # Conv indices: 0 to 2N-2.
        # For each lag j (index in conv), weight is weights[j].
        # Contribution to grad_h[k] is weights[j] * (h[j-k] + h[k-j]).
        
        # Efficient way:
        # grad_h = np.zeros(N)
        # for j in range(len(conv)):
        #     w = weights[j]
        #     if w < 1e-10: continue
        #     # d conv[j] / d h[k]
        #     # conv[j] = sum_m h[m] h[j-m]
        #     # d/dh[k] = h[j-k] (if 0<=j-k<N) + h[k-j] (if 0<=k-j<N)
        #     # Vectorized:
        #     # idx1 = j - np.arange(N)
        #     # valid1 = (idx1 >= 0) & (idx1 < N)
        #     # grad_h[valid1] += w * current_h[idx1[valid1]]
        #     # idx2 = np.arange(N) - j
        #     # valid2 = (idx2 >= 0) & (idx2 < N)
        #     # grad_h[valid2] += w * current_h[idx2[valid2]]
        
        # Let's do the loop for clarity and safety, N=50, len(conv)=99. 99*50 = 4950 ops per iter. Fast.
        for j in range(len(conv)):
            w = weights[j]
            if w < 1e-12:
                continue
            # Term 1: h[j-k]
            # k ranges 0..N-1. j-k ranges j..j-N+1.
            # Valid k if 0 <= j-k < N => j-N < k <= j.
            # k_min = max(0, j-N+1)
            # k_max = min(N-1, j)
            k_min = max(0, j - N + 1)
            k_max = min(N - 1, j)
            if k_min <= k_max:
                ks = np.arange(k_min, k_max + 1)
                idxs = j - ks
                grad_h[ks] += w * current_h[idxs]
            
            # Term 2: h[k-j]
            # Valid k if 0 <= k-j < N => j <= k < j+N.
            # k_min = max(0, j)
            # k_max = min(N-1, j+N-1)
            k_min = max(0, j)
            k_max = min(N - 1, j + N - 1)
            if k_min <= k_max:
                ks = np.arange(k_min, k_max + 1)
                idxs = ks - j
                grad_h[ks] += w * current_h[idxs]
        
        # 2. Gradient of Objective
        # We want to maximize R = L2^2 / S.
        # dR/du = (dR/dh) * (dh/du)
        # dh/du = h (since h = exp(u)/sum(exp(u)), and sum(exp(u)) is constant w.r.t u_i in the local derivative? 
        # No, h_i = exp(u_i)/Z_u. dh_i/du_i = h_i - h_i^2/Z_u? No.
        # Let's just use multiplicative update on h directly, which is equivalent to 
        # gradient ascent on log-h with appropriate scaling.
        
        # d/dh_i (L2^2) = 2 h_i
        # d/dh_i (1/S) = -1/S^2 * dS/dh_i
        # dR/dh_i = 2 h_i / S - (L2^2 / S^2) * dS/dh_i
        # dR/dh_i = (1/S) * [ 2 h_i - (L2^2 / S) * dS/dh_i ]
        
        S = M + (1.0 / beta) * np.log(Z) # The soft sup value
        L2_sq = np.sum(current_h**2)
        
        # If S is very small, gradient might be unstable.
        if S < 1e-15:
            break
            
        obj_grad_h = (2.0 * current_h - (L2_sq / S) * grad_h) / S
        
        # 3. Update
        # We want to increase h where obj_grad_h is positive.
        # Multiplicative update: h_new = h * exp(eta * obj_grad_h)
        # But we need to normalize.
        
        # To keep it stable, we can use a smaller eta.
        # Also, we should ensure we don't go to 0.
        
        # Normalizing gradient to prevent huge jumps
        grad_norm = np.linalg.norm(obj_grad_h)
        if grad_norm > 1e-15:
            obj_grad_h_norm = obj_grad_h / grad_norm
        else:
            obj_grad_h_norm = obj_grad_h
            
        # Adaptive eta
        # If we are making progress, keep eta. If not, reduce.
        # For now, fixed eta with decay if stalled.
        
        update = np.exp(eta * obj_grad_h_norm)
        # Clip update to prevent extreme changes
        update = np.clip(update, 0.5, 2.0)
        
        current_h = current_h * update
        
        # Renormalize to sum=1
        s = np.sum(current_h)
        if s < 1e-15:
            break
        current_h /= s
        
        # 4. Evaluate
        r_new = _compute_ratio(current_h)
        
        if r_new > best_ratio:
            best_ratio = r_new
            best_h = current_h.copy()
            stale = 0
            # Increase beta slightly to sharpen the soft-max as we converge
            if beta < 50.0:
                beta *= 1.05
        else:
            stale += 1
            if stale > patience:
                # Random restart or perturbation
                # Perturb log_h
                noise = np.random.randn(N) * 0.1
                log_h = np.log(current_h + 1e-30) + noise
                # Softmax to get new h
                log_h -= np.max(log_h)
                h_new = np.exp(log_h)
                h_new /= np.sum(h_new)
                current_h = h_new
                stale = 0
                beta = 5.0 # Reset beta
                eta = 0.05

    # Polishing Phase
    # The soft-max optimization might stop slightly away from the optimal hard-max solution.
    # Perform local search on the best_h found.
    if time.time() < deadline - 5.0:
        h_polish = best_h.copy()
        for _ in range(100):
            if time.time() > deadline - 2.0:
                break
            improved = False
            # Try moving mass from i to j
            # Simple coordinate descent: for each i, try to set h_i to 0 or increase it?
            # Better: Try small perturbations.
            # Given N=50, checking all pairs is 2500.
            # Let's try: for each i, reduce h_i by epsilon, add to h_j.
            # This is expensive. 
            # Alternative: Use the gradient of the *exact* objective (hard max) for a few steps.
            
            # Compute exact gradient of Hard Max
            conv = np.convolve(h_polish, h_polish, mode='full')
            sup_conv = np.max(conv)
            L2_sq = np.sum(h_polish**2)
            
            if sup_conv < 1e-15:
                break
                
            # Gradient of 1/SupConv is tricky at the max.
            # But we can just do a simple "repel from max" step.
            k_star = np.argmax(conv)
            
            grad_h = np.zeros(N)
            for i in range(N):
                val = 0.0
                idx1 = k_star - i
                if 0 <= idx1 < N:
                    val += h_polish[idx1]
                idx2 = i - k_star
                if 0 <= idx2 < N:
                    val += h_polish[idx2]
                grad_h[i] = val
                
            # Objective: Max L2^2 / Sup
            # Grad L2^2 = 2 h
            # Grad (1/Sup) ~ -1/Sup^2 * Grad Sup
            # So we want to increase h where 2h is high and Grad Sup is low.
            # And decrease h where Grad Sup is high.
            
            # Normalized gradient
            g = 2.0 * h_polish - (L2_sq / sup_conv) * grad_h / sup_conv
            g /= np.linalg.norm(g) + 1e-15
            
            # Small step
            step = 0.05
            h_new = h_polish + step * g
            h_new[h_new < 0] = 0
            s = np.sum(h_new)
            if s < 1e-15:
                break
            h_new /= s
            
            r_new = _compute_ratio(h_new)
            r_cur = _compute_ratio(h_polish)
            
            if r_new > r_cur + 1e-10:
                h_polish = h_new
                improved = True
            else:
                # If no improvement, try a smaller step
                step *= 0.5
                if step < 1e-6:
                    break
                h_new = h_polish + step * g
                h_new[h_new < 0] = 0
                s = np.sum(h_new)
                if s < 1e-15:
                    break
                h_new /= s
                r_new = _compute_ratio(h_new)
                if r_new > r_cur + 1e-10:
                    h_polish = h_new
                    improved = True
                else:
                    break
        
        final_ratio = _compute_ratio(h_polish)
        if final_ratio > best_ratio:
            best_ratio = final_ratio
            best_h = h_polish

    final_ratio = _compute_ratio(best_h)
    if final_ratio > best_ratio:
        best_ratio = final_ratio

    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(_compute_ratio(h))
