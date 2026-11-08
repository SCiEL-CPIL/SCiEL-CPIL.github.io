import time
import numpy as np
from scipy import signal

N = 50

def _compute_ratio(h):
    """Compute ||h||^2_2 / (||h||_1 * ||h*h||_inf)"""
    h = np.array(h, dtype=np.float64)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    # Autoconvolution
    conv = signal.convolve(h, h, mode='full')
    conv_inf = np.max(np.abs(conv))
    if conv_inf == 0:
        return 0.0
    return l2_sq / (l1 * conv_inf)

def construct():
    deadline = time.time() + 100.0  # Safety margin before 120s limit
    best_h = [1.0] * N  # Seed as fallback
    best_ratio = _compute_ratio(best_h)
    
    # Try various structured configurations
    candidates = []
    
    # 1. Uniform
    candidates.append([1.0] * N)
    
    # 2. Single spike
    for pos in [0, N//2, N-1]:
        h = [0.0] * N
        h[pos] = 1.0
        candidates.append(h)
    
    # 3. Two spikes at various separations
    for sep in [1, 5, 10, 15, 20, 25, 30, 40, 49]:
        if sep < N:
            h = [0.0] * N
            h[0] = 1.0
            h[sep] = 1.0
            candidates.append(h)
    
    # 4. Block of consecutive ones
    for block_len in [1, 2, 5, 10, 20, 25, 50]:
        h = [0.0] * N
        h[:block_len] = [1.0] * block_len
        candidates.append(h)
        # Also centered
        if block_len < N:
            h2 = [0.0] * N
            start = (N - block_len) // 2
            h2[start:start+block_len] = [1.0] * block_len
            candidates.append(h2)
    
    # 5. Alternating patterns
    for period in [2, 3, 4, 5, 10]:
        h = [0.0] * N
        for i in range(N):
            if i % period == 0:
                h[i] = 1.0
        candidates.append(h)
    
    # 6. Gaussian-like
    center = N // 2
    for sigma in [1, 2, 5, 10, 20]:
        h = [0.0] * N
        for i in range(N):
            h[i] = np.exp(-0.5 * ((i - center) / sigma)**2)
        candidates.append(h)
    
    # 7. Cosine window
    for freq in [1, 2, 3, 5]:
        h = [0.0] * N
        for i in range(N):
            val = np.cos(2 * np.pi * freq * i / N)
            h[i] = max(0.0, val)
        candidates.append(h)
    
    # 8. Random configurations (limited by time)
    np.random.seed(42)
    num_random = 200
    for r in range(num_random):
        if time.time() > deadline - 30.0:
            break
        # Random sparse
        num_nonzero = np.random.randint(1, 10)
        positions = np.random.choice(N, size=num_nonzero, replace=False)
        h = [0.0] * N
        for pos in positions:
            h[pos] = np.random.uniform(0.1, 1.0)
        candidates.append(h)
        
        # Random dense
        h = list(np.random.uniform(0.0, 1.0, N))
        candidates.append(h)
    
    # Evaluate all candidates
    for h in candidates:
        if time.time() > deadline - 5.0:
            break
        ratio = _compute_ratio(h)
        if ratio > best_ratio:
            best_ratio = ratio
            best_h = h
    
    return best_h
