import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    # Keep a safety margin of 20 seconds
    deadline = start_time + 100.0

    def compute_value(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        c = np.convolve(h, h)
        return abs(2.0 * N * np.max(c) / (s * s))

    h_best = np.ones(N)
    val_best = compute_value(h_best)
    
    # Seed score is 2.0, so current val_best is 2.0
    # Score = 2.0 / val. Current score 1.0.
    # To improve score, we need val < 2.0.
    
    rng = np.random.default_rng(42)

    # Inspired by the fact that the current program (score 1.1774) likely found a specific local minimum.
    # The "Inspiration 1" (score 1.1014) tried structured perturbations like sinusoids.
    # The "Inspiration 2" was just the seed.
    # The current program uses random noise + local search.
    
    # Let's combine: 
    # 1. Try structured perturbations (sinusoids, etc.) to find good starting points.
    # 2. Use local search to refine them.
    # 3. Also try random restarts with local search.
    
    # Strategy:
    # The value is scale invariant. We can fix sum(h) = N (mean 1).
    # Then value = 2 * max(conv(h,h)) / N.
    # For h = ones, conv is triangle max N, so value = 2 * N / N = 2.
    # We want to minimize max(conv(h,h)) given sum(h)=N.
    # Let h = 1 + f, where mean(f)=0.
    # conv(h,h) = conv(1,1) + conv(1,f) + conv(f,1) + conv(f,f)
    # conv(1,1) is a triangle peak at center.
    # We want to "flatten" this peak or push it down.
    # conv(1,f) + conv(f,1) is linear in f.
    # conv(f,f) is quadratic.
    
    # If we choose f to be a low frequency cosine, it might interfere destructively with the triangle peak?
    # The triangle peak is positive everywhere.
    # If we make h oscillate, the auto-correlation might have negative parts, but we take max.
    # Actually, if we make h have negative values, the sum is fixed, so positive parts must be larger.
    
    # Let's try to optimize h directly using gradient-free methods or simple perturbations.
    # The current program does random noise + hill climbing.
    # Let's add structured initializations.
    
    candidates = []
    
    # 1. Structured perturbations: h = 1 + eps * sin(2*pi*k*i/N)
    for k in range(1, 20):
        for eps in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0]:
            f = np.sin(2 * np.pi * k * np.arange(N) / N)
            f = f - np.mean(f)
            h = 1.0 + eps * f
            val = compute_value(h)
            if val < val_best:
                val_best = val
                h_best = h.copy()
            candidates.append((val, h.copy()))
            
    # 2. Cosine waves
    for k in range(1, 20):
        for eps in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0]:
            f = np.cos(2 * np.pi * k * np.arange(N) / N)
            f = f - np.mean(f)
            h = 1.0 + eps * f
            val = compute_value(h)
            if val < val_best:
                val_best = val
                h_best = h.copy()
            candidates.append((val, h.copy()))
            
    # 3. Random perturbations
    for _ in range(100):
        if time.time() > deadline:
            break
        f = rng.normal(0, 1, N)
        f = f - np.mean(f)
        f = f / (np.std(f) + 1e-12)
        for eps in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5]:
            h = 1.0 + eps * f
            val = compute_value(h)
            if val < val_best:
                val_best = val
                h_best = h.copy()
            candidates.append((val, h.copy()))

    # 4. Local optimization from best candidate found so far
    # Also try local optimization from the top few candidates
    candidates.sort(key=lambda x: x[0])
    top_candidates = candidates[:5]
    
    for val_init, h_init in top_candidates:
        if time.time() > deadline:
            break
        
        h_cur = h_init.copy()
        val_cur = val_init
        
        # Hill climbing
        for step in range(200):
            if time.time() > deadline:
                break
            improved = False
            # Try perturbing a few random indices
            for _ in range(10):
                idx = rng.integers(0, N)
                delta = rng.normal(0, 0.1)
                h_new = h_cur.copy()
                h_new[idx] += delta
                val_new = compute_value(h_new)
                if val_new < val_cur:
                    val_cur = val_new
                    h_cur = h_new
                    improved = True
            if not improved:
                # Try larger steps
                for _ in range(5):
                    idx = rng.integers(0, N)
                    delta = rng.normal(0, 0.5)
                    h_new = h_cur.copy()
                    h_new[idx] += delta
                    val_new = compute_value(h_new)
                    if val_new < val_cur:
                        val_cur = val_new
                        h_cur = h_new
                        improved = True
            if not improved:
                # Try block perturbations
                for _ in range(5):
                    start = rng.integers(0, N - 10)
                    end = start + rng.integers(5, 20)
                    delta = rng.normal(0, 0.2)
                    h_new = h_cur.copy()
                    h_new[start:end] += delta
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
