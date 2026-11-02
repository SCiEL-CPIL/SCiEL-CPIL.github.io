import numpy as np
import time
import random

N = 50
DEADLINE = 145  # seconds, leave margin under 150s

def _score(h):
    """Compute R = ||h||_2^2 / (||h||_1 * sup|h*h|)."""
    h = np.asarray(h, dtype=np.float64)
    if h.sum() == 0:
        return 0.0
    # Autoconvolution (full)
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    l1 = np.sum(h)
    l2_sq = np.sum(h**2)
    if l1 <= 0 or sup_conv <= 0:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def _coordinate_ascent(h, max_iters=200, tol=1e-9):
    """Coordinate ascent to maximize R for nonnegative h."""
    h = np.array(h, dtype=np.float64, copy=True)
    h[h < 0] = 0.0
    if h.sum() == 0:
        h[0] = 1.0
    best_score = _score(h)
    for it in range(max_iters):
        improved = False
        for i in range(N):
            if h[i] < 1e-15:
                # Try setting to a small value
                trial = h.copy()
                trial[i] = 1e-3
                s = _score(trial)
                if s > best_score + tol:
                    h = trial
                    best_score = s
                    improved = True
                continue
            # Optimize h[i] by 1D search
            # R as function of x = h[i]:
            # ||h||_2^2 = x^2 + rest2
            # ||h||_1 = x + rest1
            # sup|h*h| = max over k of (x*h_{i+k} + x*h_{i-k} + rest_conv_k)
            # This is complex; do a simple golden-section or grid search
            rest2 = np.sum(h**2) - h[i]**2
            rest1 = np.sum(h) - h[i]
            
            # Try a grid of multipliers around current value
            best_x = h[i]
            best_s = best_score
            for mult in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 3.0, 5.0]:
                trial_x = h[i] * mult
                if trial_x < 0:
                    continue
                trial = h.copy()
                trial[i] = trial_x
                s = _score(trial)
                if s > best_s:
                    best_s = s
                    best_x = trial_x
            if abs(best_x - h[i]) > tol:
                h[i] = best_x
                best_score = best_s
                improved = True
        if not improved:
            break
    return h, best_score

def _sidon_set_search(max_iters=5000):
    """Search for Sidon-like binary patterns with good ratio."""
    best_h = None
    best_s = 0.0
    for _ in range(max_iters):
        # Random binary vector with density ~0.3
        density = random.uniform(0.2, 0.5)
        h = (np.random.rand(N) < density).astype(float)
        if h.sum() == 0:
            h[0] = 1.0
        s = _score(h)
        if s > best_s:
            best_s = s
            best_h = h.copy()
    return best_h, best_s

def _gaussian_like(center, width):
    """Generate a smooth Gaussian-like bump."""
    x = np.arange(N)
    h = np.exp(-0.5 * ((x - center) / width) ** 2)
    h[h < 1e-10] = 0.0
    return h

def construct():
    start_time = time.time()
    
    # Initial candidates
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones(N) * 0.01)
    
    # 2. Spike
    for pos in range(N):
        h = np.zeros(N)
        h[pos] = 1.0
        candidates.append(h)
    
    # 3. Gaussian bumps at various centers/widths
    for center in range(0, N, 5):
        for width in [2, 3, 5, 8]:
            h = _gaussian_like(center, width)
            if h.sum() > 0:
                candidates.append(h * 0.01)
    
    # 4. Random
    for _ in range(20):
        h = np.random.rand(N)
        h *= 0.01
        candidates.append(h)
    
    # 5. Sidon-like binary
    sidon_h, sidon_s = _sidon_set_search(max_iters=2000)
    if sidon_h is not None:
        candidates.append(sidon_h * 0.01)
    
    # Evaluate all candidates
    best_h = None
    best_score = 0.0
    for h in candidates:
        h = np.asarray(h, dtype=np.float64)
        h[h < 0] = 0.0
        if h.sum() == 0:
            h[0] = 1.0
        s = _score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Refine with coordinate ascent
    if time.time() - start_time < DEADLINE - 30:
        best_h, best_score = _coordinate_ascent(best_h, max_iters=300)
    
    # Try multiple starting points with coordinate ascent
    if time.time() - start_time < DEADLINE - 20:
        extra_starts = [
            np.ones(N) * 0.01,
            _gaussian_like(N//2, 3) * 0.01,
            _gaussian_like(10, 2) * 0.01,
            np.random.rand(N) * 0.01,
        ]
        for sh in extra_starts:
            if time.time() - start_time > DEADLINE - 15:
                break
            rh, rs = _coordinate_ascent(sh, max_iters=150)
            if rs > best_score:
                best_score = rs
                best_h = rh
    
    # Final safety: ensure nonnegative
    best_h = np.asarray(best_h, dtype=np.float64)
    best_h[best_h < 0] = 0.0
    if best_h.sum() == 0:
        best_h = [0.01] * N
    else:
        best_h = list(best_h)
    
    return best_h

if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    print(f"Time: {time.time() - t0:.2f}s, N={len(result)}")
    print(f"Score: {_score(result):.6f}")
    print(f"First 10: {result[:10]}")
