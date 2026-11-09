"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

def _compute_ratio(h):
    """Compute the ratio: sum(h^2) / (sum(h) * max(autoconv))"""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-15):
        return -1.0
    h = np.maximum(h, 0.0)
    l2_sq = np.sum(h * h)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    # Autoconvolution via FFT or direct
    auto = np.convolve(h, h, mode='full')
    max_auto = np.max(auto)
    if max_auto < 1e-15:
        return 0.0
    return l2_sq / (l1 * max_auto)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_h = [1.0] * N
    best_val = _compute_ratio(best_h)
    
    # Try several structural families
    candidates = []
    
    # 1. Single spike at various positions
    for pos in range(N):
        h = [0.0] * N
        h[pos] = 1.0
        candidates.append(h)
    
    # 2. Two adjacent spikes
    for pos in range(N - 1):
        h = [0.0] * N
        h[pos] = 1.0
        h[pos + 1] = 1.0
        candidates.append(h)
    
    # 3. Block of k consecutive spikes
    for k in [2, 3, 5, 10, 25]:
        for pos in range(0, N - k + 1, max(1, (N - k) // 5)):
            h = [0.0] * N
            for i in range(pos, pos + k):
                h[i] = 1.0
            candidates.append(h)
    
    # 4. Gaussian-like concentration
    center = N / 2.0
    for sigma in [1.0, 2.0, 3.0, 5.0, 10.0]:
        h = []
        for i in range(N):
            h.append(np.exp(-0.5 * ((i - center) / sigma) ** 2))
        candidates.append(h)
    
    # 5. Cosine window
    for width in [5, 10, 20, 50]:
        h = []
        for i in range(N):
            if abs(i - center) <= width / 2:
                h.append(0.5 * (1 + np.cos(2 * np.pi * (i - center) / width)))
            else:
                h.append(0.0)
        candidates.append(h)
    
    # 6. Triangle peak
    for peak in [1, 2, 5, 10]:
        h = []
        for i in range(N):
            dist = abs(i - center)
            val = max(0, peak - dist)
            h.append(val)
        candidates.append(h)
    
    # 7. Random sparse profiles
    np.random.seed(42)
    for trial in range(200):
        n_nonzero = np.random.randint(1, N + 1)
        idx = np.random.choice(N, size=n_nonzero, replace=False)
        h = [0.0] * N
        for i in idx:
            h[i] = 1.0
        candidates.append(h)
    
    # 8. Random dense profiles with varying magnitudes
    for trial in range(100):
        h = np.random.rand(N)
        # Apply some structure: keep top-k
        k = np.random.randint(1, N + 1)
        thresh = np.sort(h)[-k]
        h = np.where(h >= thresh, h, 0.0)
        candidates.append(list(h))
    
    # Evaluate all candidates
    for h in candidates:
        if time.time() > deadline:
            break
        val = _compute_ratio(h)
        if val > best_val:
            best_val = val
            best_h = list(h)
    
    # Local search: perturb the best solution
    if time.time() < deadline - 5:
        current_h = np.array(best_h, dtype=np.float64)
        current_val = best_val
        step_size = 0.5
        for iteration in range(5000):
            if time.time() > deadline:
                break
            # Random perturbation
            idx = np.random.randint(N)
            delta = np.random.randn() * step_size
            new_h = current_h.copy()
            new_h[idx] = max(0.0, new_h[idx] + delta)
            val = _compute_ratio(new_h)
            if val > current_val:
                current_val = val
                current_h = new_h
                best_h = list(current_h)
                best_val = current_val
                # Occasionally reduce step size
                if iteration % 1000 == 0:
                    step_size *= 0.8
            else:
                # Occasionally try larger jumps
                if np.random.random() < 0.01:
                    step_size = 0.5
    
    return best_h
# EVOLVE-BLOCK-END
