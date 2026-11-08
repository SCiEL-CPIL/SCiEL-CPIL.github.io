import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 145 # 5s margin
    
    def compute_score(h):
        h = np.asarray(h, dtype=float)
        l2_sq = np.sum(h**2)
        if l2_sq < 1e-15:
            return float('inf'), 0, np.zeros(N)
        h_conv = np.convolve(h, h, mode='full')
        max_val = np.max(h_conv)
        max_idx = np.argmax(h_conv)
        score = max_val / l2_sq
        return score, max_idx, h_conv

    best_h = np.ones(N)
    best_score, _, _ = compute_score(best_h)
    
    # Multiple restarts
    for restart in range(20):
        if time.time() > deadline - 10:
            break
            
        # Initialize
        # Try different initializations
        if restart % 3 == 0:
            # Gaussian
            x = np.linspace(-1, 1, N)
            h = np.exp(-x**2 / (2 * 0.1**2))
        elif restart % 3 == 1:
            # Two bumps
            x = np.linspace(-1, 1, N)
            h = np.exp(-((x-0.3)**2)/0.01) + np.exp(-((x+0.3)**2)/0.01)
        else:
            # Random
            h = np.random.rand(N)
            
        h = np.clip(h, 0, None)
        
        # Gradient Descent
        lr = 0.01
        momentum = 0.9
        vel = np.zeros_like(h)
        
        for step in range(5000):
            if time.time() > deadline - 5:
                break
                
            score, max_idx, h_conv = compute_score(h)
            
            if score < best_score:
                best_score = score
                best_h = h.copy()
                
            l2_sq = np.sum(h**2)
            if l2_sq < 1e-15: break
            
            # Gradient of N = h_conv[max_idx]
            # h_conv[k] = sum_i h[i] h[k-i]
            # d/dh[j] h_conv[k] = h[k-j] + h[k-j] ? No.
            # h_conv[k] = sum_i h[i] h[k-i]
            # Term with h[j]: h[j] h[k-j] + h[k-j] h[j] = 2 h[j] h[k-j] ?
            # Wait.
            # If i=j, term is h[j] h[k-j].
            # If k-i=j => i=k-j, term is h[k-j] h[j].
            # So if j != k-j, we have 2 h[j] h[k-j].
            # If j == k-j (i.e., 2j=k), we have h[j]^2, derivative is 2h[j].
            # Generally, grad[j] = 2 * h[(k-j)] if 0 <= k-j < N.
            # Note: if k-j is out of bounds, h[k-j] is 0.
            
            grad_N = np.zeros(N)
            # Indices j such that 0 <= k-j < N
            # j ranges from max(0, k-(N-1)) to min(N-1, k)
            start_j = max(0, max_idx - (N - 1))
            end_j = min(N - 1, max_idx)
            
            # h[k-j] for j in [start_j, end_j]
            # Indices for h: max_idx - j
            # j = start_j -> idx = max_idx - start_j
            # j = end_j -> idx = max_idx - end_j
            # Since j increases, idx decreases.
            
            idxs = max_idx - np.arange(start_j, end_j + 1)
            grad_N[start_j:end_j+1] = 2 * h[idxs]
            
            # Gradient of D = sum(h^2)
            grad_D = 2 * h
            
            # Gradient of Score = N/D
            grad_score = (grad_N * l2_sq - N * grad_D) / (l2_sq ** 2)
            
            # Update
            vel = momentum * vel - lr * grad_score
            h_new = h + vel
            h = np.clip(h_new, 0, None)
            
            # Adaptive LR
            if step % 100 == 0:
                lr *= 0.95

    return list(best_h)
