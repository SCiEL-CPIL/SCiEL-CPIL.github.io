import numpy as np
import time

N = 600

def compute_score(heights):
    """Compute the normalized max of autoconvolution."""
    f = np.array(heights, dtype=float)
    if np.any(f < -1e-15):
        return float('inf')
    f = np.clip(f, 0, None)
    s = np.sum(f)
    if s == 0:
        return float('inf')
    # Compute autoconvolution
    autoconv = np.convolve(f, f, mode='full')
    return np.max(autoconv) / s

def construct():
    start = time.time()
    deadline = start + 140  # 10s safety margin
    
    best_score = float('inf')
    best_heights = [1.0] * N
    
    # Try various window shapes
    x = np.linspace(0, 1, N)
    
    # 1. Uniform (seed)
    heights = np.ones(N)
    score = compute_score(heights)
    if score < best_score:
        best_score = score
        best_heights = heights.tolist()
    
    # 2. Cosine window (tapered at edges)
    heights = 0.5 * (1 + np.cos(2 * np.pi * x))
    score = compute_score(heights)
    if score < best_score:
        best_score = score
        best_heights = heights.tolist()
    
    # 3. Hamming window
    heights = 0.54 - 0.46 * np.cos(2 * np.pi * x)
    score = compute_score(heights)
    if score < best_score:
        best_score = score
        best_heights = heights.tolist()
    
    # 4. Blackman window
    heights = 0.42 - 0.5 * np.cos(2 * np.pi * x) + 0.08 * np.cos(4 * np.pi * x)
    score = compute_score(heights)
    if score < best_score:
        best_score = score
        best_heights = heights.tolist()
    
    # 5. Inverse cosine (peak in center)
    heights = np.cos(np.pi * x)
    heights = np.clip(heights, 0, None)
    score = compute_score(heights)
    if score < best_score:
        best_score = score
        best_heights = heights.tolist()
    
    # 6. Try to minimize via gradient descent on a smooth surrogate
    # Use log-sum-exp as soft max
    def soft_max_autoconv(f, temperature=0.1):
        autoconv = np.convolve(f, f, mode='full')
        # Soft max
        m = np.max(autoconv)
        if m == 0:
            return 0
        shifted = autoconv / temperature
        shifted -= np.max(shifted)
        logsumexp = np.log(np.sum(np.exp(shifted))) + np.max(shifted)
        return logsumexp * temperature
    
    # Gradient descent
    if time.time() < deadline:
        f = np.ones(N)
        lr = 0.01
        for step in range(5000):
            if time.time() > deadline:
                break
            # Compute gradient of max(autoconv)/sum(f)
            autoconv = np.convolve(f, f, mode='full')
            s = np.sum(f)
            # Gradient of max(autoconv) w.r.t. f
            # d(autoconv[k])/df[i] = f[i+k] + f[i-k] (with appropriate bounds)
            # For the max, we use the subgradient at the argmax
            k_max = np.argmax(autoconv)
            grad_autoconv = np.zeros(N)
            for i in range(N):
                # Contribution to autoconv[k_max] from f[i]
                j1 = k_max - i
                j2 = i - k_max
                if 0 <= j1 < N:
                    grad_autoconv[i] += f[j1]
                if 0 <= j2 < N:
                    grad_autoconv[i] += f[j2]
            # Gradient of ratio max_autoconv / sum(f)
            max_val = np.max(autoconv)
            grad = (grad_autoconv * s - max_val) / (s * s)
            # Update
            f = f - lr * grad
            f = np.clip(f, 0, None)
            
            score = compute_score(f)
            if score < best_score:
                best_score = score
                best_heights = f.tolist()
    
    return best_heights
