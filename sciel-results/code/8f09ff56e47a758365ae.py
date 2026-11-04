import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    def compute_value(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        c = np.convolve(h, h)
        return abs(2.0 * N * np.max(c) / (s * s))
    
    h_best = np.ones(N)
    val_best = compute_value(h_best)
    
    rng = np.random.default_rng(42)
    
    # Try structured perturbations
    # Strategy: h = 1 + ε * f, where f has zero mean
    # This keeps sum = N, so value = 2N * max(conv(1+εf, 1+εf)) / N²
    # = 2 * max(conv(1+εf, 1+εf)) / N
    
    # conv(1+εf, 1+εf) = conv(1,1) + ε(conv(1,f)+conv(f,1)) + ε² conv(f,f)
    # conv(1,1) is triangle with max N
    # We want to minimize the max of the whole thing
    
    # Try various f: sinusoids, random, etc.
    
    candidates = []
    
    # 1. Pure sinusoids
    for k in range(1, 50):
        for eps in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]:
            f = np.sin(2 * np.pi * k * np.arange(N) / N)
            f = f - np.mean(f)  # zero mean
            h = 1.0 + eps * f
            val = compute_value(h)
            if val < val_best:
                val_best = val
                h_best = h.copy()
            candidates.append((val, h.copy()))
    
    # 2. Cosine waves
    for k in range(1, 50):
        for eps in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]:
            f = np.cos(2 * np.pi * k * np.arange(N) / N)
            f = f - np.mean(f)
            h = 1.0 + eps * f
            val = compute_value(h)
            if val < val_best:
                val_best = val
                h_best = h.copy()
            candidates.append((val, h.copy()))
    
    # 3. Alternating
    f = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    f = f - np.mean(f)
    for eps in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]:
        h = 1.0 + eps * f
        val = compute_value(h)
        if val < val_best:
            val_best = val
            h_best = h.copy()
        candidates.append((val, h.copy()))
    
    # 4. Random perturbations
    for _ in range(200):
        if time.time() > deadline:
            break
        f = rng.normal(0, 1, N)
        f = f - np.mean(f)
        f = f / (np.std(f) + 1e-12)
        for eps in [0.1, 0.3, 0.5, 0.7, 0.9]:
            h = 1.0 + eps * f
            val = compute_value(h)
            if val < val_best:
                val_best = val
                h_best = h.copy()
            candidates.append((val, h.copy()))
    
    # 5. Local optimization from best candidate
    if time.time() < deadline:
        h_cur = h_best.copy()
        val_cur = val_best
        for step in range(500):
            if time.time() > deadline:
                break
            improved = False
            for _ in range(5):
                idx = rng.integers(0, N)
                delta = rng.normal(0, 0.05)
                h_new = h_cur.copy()
                h_new[idx] += delta
                val_new = compute_value(h_new)
                if val_new < val_cur:
                    val_cur = val_new
                    h_cur = h_new
                    improved = True
            if not improved:
                # Try larger steps
                for _ in range(3):
                    idx = rng.integers(0, N)
                    delta = rng.normal(0, 0.2)
                    h_new = h_cur.copy()
                    h_new[idx] += delta
                    val_new = compute_value(h_new)
                    if val_new < val_cur:
                        val_cur = val_new
                        h_cur = h_new
                        improved = True
            if not improved:
                break
        
        if val_cur < val_best:
            val_best = val_cur
            h_best = h_cur
    
    return h_best
