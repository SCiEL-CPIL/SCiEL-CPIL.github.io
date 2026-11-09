"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 50

# EVOLVE-BLOCK-START
import time

def _score(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for non-negative h."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.dot(h, h)
    # Full linear convolution (autoconvolution)
    conv = np.convolve(h, h, mode='full')
    sup = np.max(conv)
    if sup == 0:
        return 0.0
    return l2_sq / (l1 * sup)

def construct():
    """Return 50 nonnegative step heights maximizing the autocorrelation ratio.
    
    The ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) is provably <= 1 for any
    non-negative h, with equality when (h*h)_0 = ||h||_2^2, i.e., when all
    mass is at one point (delta) or more generally when the convolution peak
    equals the L2 norm squared. We verify this by trying several candidate
    shapes and returning the best one found, with a time guard.
    """
    t_start = time.time()
    deadline = t_start + 110.0  # 10s safety margin before 120s limit
    
    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_score = _score(best_h)
    
    # Try a variety of candidate shapes
    candidates = []
    
    # 1. Delta at various positions
    for i in range(N):
        h = np.zeros(N)
        h[i] = 1.0
        candidates.append(h)
    
    # 2. Uniform
    h = np.ones(N)
    candidates.append(h)
    
    # 3. Gaussian-like shapes (various widths and centers)
    x = np.arange(N)
    for center in [0, N//4, N//2, 3*N//4, N-1]:
        for sigma in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
            h = np.exp(-0.5 * ((x - center) / sigma) ** 2)
            if np.max(h) > 0:
                h = h / np.sum(h)  # normalize L1=1
                candidates.append(h)
    
    # 4. Triangular shapes
    for center in [0, N//2, N-1]:
        h = np.maximum(0, 1 - np.abs(x - center) / (N // 2))
        if np.sum(h) > 0:
            h = h / np.sum(h)
            candidates.append(h)
    
    # 5. Power-law shapes
    for center in [0, N//2]:
        for p in [0.5, 1.0, 2.0, 4.0, 8.0]:
            h = np.power(np.maximum(0, 1 - np.abs(x - center) / (N // 2)), p)
            if np.sum(h) > 0:
                h = h / np.sum(h)
                candidates.append(h)
    
    # 6. Two-peak shapes
    for gap in [1, 2, 5, 10, 20]:
        for c in [0, N//2 - gap//2, N - gap - 1]:
            h = np.zeros(N)
            if 0 <= c < N and 0 <= c + gap < N:
                h[c] = 0.5
                h[c + gap] = 0.5
            if np.sum(h) > 0:
                candidates.append(h)
    
    # 7. Random non-negative vectors (few, fast)
    rng = np.random.default_rng(42)
    for _ in range(200):
        h = rng.random(N)
        if np.sum(h) > 0:
            h = h / np.sum(h)
            candidates.append(h)
        if time.time() > deadline:
            break
    
    # Evaluate all candidates
    for h in candidates:
        if time.time() > deadline:
            break
        s = _score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # If we have time, try a simple local search from the best candidate
    if time.time() < deadline - 10:
        # Coordinate ascent: perturb each coordinate slightly
        current = best_h.copy()
        current_score = best_score
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(N):
                for delta_frac in [0.1, 0.3, 0.5, 0.9]:
                    trial = current.copy()
                    # Move mass from i to i+1 (or redistribute)
                    if i < N - 1:
                        trial[i] = current[i] * (1 - delta_frac)
                        trial[i+1] = current[i+1] + current[i] * delta_frac
                    elif i == N - 1:
                        trial[i] = current[i] * (1 - delta_frac)
                        trial[i-1] = current[i-1] + current[i] * delta_frac
                    # Ensure non-negativity
                    if np.any(trial < -1e-15):
                        continue
                    trial = np.maximum(trial, 0)
                    if np.sum(trial) == 0:
                        continue
                    trial = trial / np.sum(trial)
                    s = _score(trial)
                    if s > current_score + 1e-12:
                        current = trial
                        current_score = s
                        best_score = s
                        best_h = trial.copy()
                        improved = True
                        break
    
    return list(best_h)
# EVOLVE-BLOCK-END
