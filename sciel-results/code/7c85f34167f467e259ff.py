"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def compute_ratio(h):
        """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
        h = np.asarray(h, dtype=float)
        if np.any(h < 0):
            h = np.maximum(h, 0)
        l1 = np.sum(h)
        if l1 == 0:
            return 0.0
        l2_sq = np.sum(h**2)
        # Autoconvolution: (h*h)[k] = sum_i h[i] * h[i+k]
        # Using FFT for efficiency
        h_pad = np.zeros(2 * N - 1)
        h_pad[:N] = h
        conv = np.fft.irfft(np.fft.rfft(h_pad) ** 2)
        conv_inf = np.max(np.abs(conv))
        if conv_inf == 0:
            return 0.0
        return l2_sq / (l1 * conv_inf)
    
    def compute_ratio_grad(h):
        """Compute ratio and gradient via finite differences (for small N)"""
        h = np.asarray(h, dtype=float)
        base_ratio = compute_ratio(h)
        grad = np.zeros_like(h)
        eps = 1e-7
        for i in range(N):
            h_pert = h.copy()
            h_pert[i] += eps
            grad[i] = (compute_ratio(h_pert) - base_ratio) / eps
        return base_ratio, grad
    
    # Try several initial patterns and optimize the best
    candidates = []
    
    # 1. Single spike
    h1 = np.zeros(N)
    h1[0] = 1.0
    candidates.append(h1)
    
    # 2. Two spikes far apart
    h2 = np.zeros(N)
    h2[0] = 1.0
    h2[N//2] = 1.0
    candidates.append(h2)
    
    # 3. Sparse pattern (every other element)
    h3 = np.zeros(N)
    h3[::2] = 1.0
    candidates.append(h3)
    
    # 4. Random sparse patterns
    rng = np.random.RandomState(42)
    for _ in range(20):
        h_r = np.zeros(N)
        n_spikes = rng.randint(1, 10)
        positions = rng.choice(N, size=n_spikes, replace=False)
        h_r[positions] = 1.0
        candidates.append(h_r)
    
    # 5. Geometric patterns
    for start in [0, 1, 2]:
        h_g = np.zeros(N)
        for i in range(N):
            h_g[i] = (0.5 ** i) if i >= start else 0
        if np.sum(h_g) > 0:
            candidates.append(h_g)
    
    # Evaluate all candidates
    best_h = None
    best_ratio = 0.0
    for h_c in candidates:
        r = compute_ratio(h_c)
        if r > best_ratio:
            best_ratio = r
            best_h = h_c.copy()
    
    # Gradient ascent from the best candidate
    if best_h is not None:
        h_opt = best_h.copy()
        lr = 0.1
        for step in range(200):
            if time.time() > deadline:
                break
            ratio, grad = compute_ratio_grad(h_opt)
            if ratio > best_ratio:
                best_ratio = ratio
                best_h = h_opt.copy()
            # Gradient ascent with projection onto nonnegativity
            h_new = h_opt + lr * grad
            h_new = np.maximum(h_new, 0)
            # Normalize to avoid degenerate scaling (ratio is scale-invariant)
            if np.sum(h_new) > 0:
                h_new = h_new / np.sum(h_new)
            # Check improvement
            r_new = compute_ratio(h_new)
            if r_new > ratio + 1e-10:
                h_opt = h_new
            else:
                lr *= 0.9
                if lr < 1e-6:
                    break
    
    # Try a few more random restarts with optimization
    rng2 = np.random.RandomState(123)
    for restart in range(10):
        if time.time() > deadline:
            break
        h_r = np.zeros(N)
        n_spikes = rng2.randint(1, 15)
        positions = rng2.choice(N, size=n_spikes, replace=False)
        h_r[positions] = rng2.uniform(0.5, 1.5, size=n_spikes)
        r_r = compute_ratio(h_r)
        if r_r > best_ratio:
            best_ratio = r_r
            best_h = h_r.copy()
        
        # Quick gradient ascent
        h_opt = h_r.copy()
        lr = 0.05
        for step in range(50):
            if time.time() > deadline:
                break
            ratio, grad = compute_ratio_grad(h_opt)
            if ratio > best_ratio:
                best_ratio = ratio
                best_h = h_opt.copy()
            h_new = h_opt + lr * grad
            h_new = np.maximum(h_new, 0)
            if np.sum(h_new) > 0:
                h_new = h_new / np.sum(h_new)
            r_new = compute_ratio(h_new)
            if r_new > ratio + 1e-10:
                h_opt = h_new
            else:
                lr *= 0.8
                if lr < 1e-7:
                    break
    
    if best_h is None:
        best_h = np.ones(N)
    
    return [float(x) for x in best_h]
# EVOLVE-BLOCK-END
