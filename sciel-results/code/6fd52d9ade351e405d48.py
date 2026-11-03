import numpy as np
import time

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv <= 1e-15:
        return 0.0
    return l2_sq / (l1 * DELTA * max_conv)

def construct():
    start_time = time.time()
    deadline = 145.0
    
    def time_left():
        return deadline - time.time()
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    rng = np.random.RandomState(42)
    
    def local_search(h, max_iters=50):
        nonlocal best_score, best_h
        h = h.copy()
        for it in range(max_iters):
            if time_left() < 5:
                break
            improved = False
            # Randomly shuffle pair order
            pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
            rng.shuffle(pairs)
            for i, j in pairs:
                if time_left() < 5:
                    break
                if h[i] < 1e-15 and h[j] < 1e-15:
                    continue
                # Try moving mass both directions
                for frac in [0.1, 0.3, 0.7]:
                    move = h[i] * frac
                    if move > 1e-15:
                        h_new = h.copy()
                        h_new[i] -= move
                        h_new[j] += move
                        s = _score_raw(h_new)
                        if s > best_score + 1e-12:
                            h = h_new
                            best_score = s
                            best_h = h.copy()
                            improved = True
                            break
                    move = h[j] * frac
                    if move > 1e-15:
                        h_new = h.copy()
                        h_new[j] -= move
                        h_new[i] += move
                        s = _score_raw(h_new)
                        if s > best_score + 1e-12:
                            h = h_new
                            best_score = s
                            best_h = h.copy()
                            improved = True
                            break
            if not improved:
                break
        return h
    
    # Initial candidates
    candidates = [np.ones(N) / N]
    
    # Triangular
    tri = np.abs(np.arange(N) - N/2)
    tri = (N - tri) ** 2
    candidates.append(tri / tri.sum())
    
    # Cosine
    x = np.linspace(-1, 1, N)
    cos_b = np.clip(np.cos(np.pi * x / 2) ** 2, 0, None)
    if cos_b.sum() > 0:
        candidates.append(cos_b / cos_b.sum())
    
    # Bimodal
    bim = np.zeros(N)
    bim[:N//4] = 1
    bim[3*N//4:] = 1
    candidates.append(bim / bim.sum())
    
    # Random
    for _ in range(30):
        r = rng.random(N)
        r = np.clip(r, 0, None)
        if r.sum() > 0:
            candidates.append(r / r.sum())
    
    for c in candidates:
        if time_left() < 10:
            break
        local_search(c)
    
    # More random restarts
    while time_left() > 15:
        r = rng.random(N)
        r = np.clip(r, 0, None)
        if r.sum() <= 0:
            continue
        r = r / r.sum()
        local_search(r, max_iters=30)
    
    best_h = np.clip(best_h, 0, None)
    if best_h.sum() > 1e-15:
        best_h = best_h / best_h.sum()
    
    return list(best_h)
