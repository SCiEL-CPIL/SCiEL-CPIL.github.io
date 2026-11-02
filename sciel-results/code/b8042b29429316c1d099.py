import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def compute_ratio(h):
        if np.max(h) < 1e-15:
            return 0.0
        h_norm = h / np.max(h)
        l2_sq = np.sum(h_norm**2)
        l1 = np.sum(h_norm)
        l_inf = np.max(h_norm)
        if l1 < 1e-15:
            return 0.0
        h_conv = np.convolve(h_norm, h_norm, mode='full')
        max_conv = np.max(h_conv)
        if max_conv < 1e-15:
            return 0.0
        return (l2_sq) / (l1 * l_inf * max_conv)
    
    def gradient_step(h, lr=0.05):
        """Game-theoretic gradient flow: minimize autoconv peak relative to energy."""
        if np.max(h) < 1e-15:
            return h
        
        h_norm = h / np.max(h)
        l2_sq = np.sum(h_norm**2)
        l1 = np.sum(h_norm)
        h_conv = np.convolve(h_norm, h_norm, mode='full')
        
        # Find where max occurs
        max_idx = np.argmax(h_conv)
        max_conv = h_conv[max_idx]
        
        if max_conv < 1e-15:
            return h
        
        # Gradient of max_conv w.r.t. h_norm
        # h_conv[k] = sum_i h[i] * h[k-i]
        # d(h_conv[max_idx])/d(h[j]) = sum_i (h[i] * delta(max_idx - j - i) + h[max_idx - i] * delta(i - j))
        # = h[max_idx - j] + h[j] (when valid)
        
        grad = np.zeros(N)
        for j in range(N):
            # Contribution from h[j] in first term
            i1 = max_idx - j
            if 0 <= i1 < N:
                grad[j] += h_norm[i1]
            # Contribution from h[j] in second term
            i2 = max_idx - j
            if 0 <= i2 < N:
                grad[j] += h_norm[i2]
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-15:
            grad = grad / grad_norm
        
        # Update: decrease h where gradient is positive (reduces peak)
        h_new = h_norm - lr * grad
        h_new = np.clip(h_new, 0, None)
        
        if np.max(h_new) < 1e-15:
            return h
        
        return h_new / np.max(h_new)
    
    # Initialize with sparse pattern
    h = np.zeros(N)
    h[0] = 1.0
    h[10] = 0.8
    h[25] = 0.6
    
    best_h = h.copy()
    best_ratio = compute_ratio(best_h)
    
    # Gradient flow with annealing
    lr = 0.1
    for step in range(5000):
        if time.time() > deadline - 5.0:
            break
        
        h = gradient_step(h, lr)
        
        # Occasionally add noise to escape local minima
        if step % 100 == 0:
            noise = np.random.normal(0, 0.01, N)
            h = h + noise
            h = np.clip(h, 0, None)
            if np.max(h) > 0:
                h = h / np.max(h)
        
        ratio = compute_ratio(h)
        if ratio > best_ratio:
            best_ratio = ratio
            best_h = h.copy()
        
        # Anneal learning rate
        lr = max(0.01, lr * 0.999)
    
    return list(best_h)
