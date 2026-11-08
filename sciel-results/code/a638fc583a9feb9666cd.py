import numpy as np
import time

N = 50
DEADLINE = 140.0  # safety margin before 150s limit

def _compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for discrete heights."""
    h = np.asarray(h, dtype=float)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h ** 2)
    # Autoconvolution: (h*h)[k] = sum_i h[i] * h[i+k] for k in -(N-1)..N-1
    # Use FFT for efficiency
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * np.conj(H), n=n_fft)
    # conv[k] for k=0..N-1 gives sum_i h[i]*h[i+k]
    # conv[N-k] gives sum_i h[i]*h[i-k] = same as above by symmetry
    peak = np.max(conv)
    if peak == 0:
        return 0.0
    return l2_sq / (l1 * peak)

def construct():
    start_time = time.time()
    
    # Strategy: try various cluster patterns and optimize weights
    best_h = [1.0] * N
    best_ratio = _compute_ratio(best_h)
    
    # Try different patterns:
    # 1. Uniform (seed)
    # 2. Single cluster of size m with optimized weights
    # 3. Multiple clusters
    # 4. Alternating patterns
    
    def try_pattern(h):
        nonlocal best_h, best_ratio
        r = _compute_ratio(h)
        if r > best_ratio:
            best_ratio = r
            best_h = list(h)
    
    # Try uniform blocks of various sizes
    for m in range(1, N + 1):
        for offset in range(0, N - m + 1, max(1, (N - m) // 10)):
            h = np.zeros(N)
            h[offset:offset + m] = 1.0
            try_pattern(h)
    
    if time.time() - start_time > DEADLINE:
        return best_h
    
    # Try 2-point clusters with varying weight ratios
    for m in [2, 3, 4, 5]:
        for offset in range(0, N - m + 1, max(1, (N - m) // 5)):
            if m == 2:
                for w1 in np.linspace(0.1, 0.9, 9):
                    h = np.zeros(N)
                    h[offset] = w1
                    h[offset + 1] = 1.0 - w1
                    try_pattern(h)
            elif m == 3:
                for w1 in np.linspace(0.1, 0.6, 6):
                    for w2 in np.linspace(0.1, 0.6, 6):
                        w3 = 1.0 - w1 - w2
                        if w3 < 0.05:
                            continue
                        h = np.zeros(N)
                        h[offset] = w1
                        h[offset + 1] = w2
                        h[offset + 2] = w3
                        try_pattern(h)
            elif m == 4:
                for w1 in np.linspace(0.1, 0.5, 5):
                    for w2 in np.linspace(0.1, 0.5, 5):
                        w3 = 1.0 - w1 - w2
                        if w3 < 0.1:
                            continue
                        h = np.zeros(N)
                        h[offset] = w1
                        h[offset + 1] = w2
                        h[offset + 2] = w3
                        h[offset + 3] = 0.0
                        try_pattern(h)
    
    if time.time() - start_time > DEADLINE:
        return best_h
    
    # Try spreading mass over many points with small values
    for n_active in [10, 20, 25, 30, 40, 50]:
        h = np.zeros(N)
        indices = np.linspace(0, N - 1, n_active).astype(int)
        h[indices] = 1.0 / n_active
        try_pattern(h)
    
    if time.time() - start_time > DEADLINE:
        return best_h
    
    # Try alternating high-low patterns
    for period in [2, 3, 4, 5]:
        for phase in range(period):
            h = np.zeros(N)
            for i in range(N):
                if i % period == phase:
                    h[i] = 1.0
            try_pattern(h)
    
    if time.time() - start_time > DEADLINE:
        return best_h
    
    # Try a "comb" pattern: non-zero at every k-th position
    for k in [2, 3, 4, 5]:
        h = np.zeros(N)
        h[::k] = 1.0
        try_pattern(h)
    
    if time.time() - start_time > DEADLINE:
        return best_h
    
    # Try to optimize the best pattern found so far with local search
    # Gradient-free: perturb each element
    current = np.array(best_h, dtype=float)
    for _ in range(200):
        if time.time() - start_time > DEADLINE:
            break
        improved = False
        for i in range(N):
            for delta in [0.01, -0.01, 0.05, -0.05, 0.1, -0.1]:
                trial = current.copy()
                trial[i] = max(0.0, trial[i] + delta)
                r = _compute_ratio(trial)
                if r > best_ratio + 1e-12:
                    best_ratio = r
                    best_h = list(trial)
                    current = trial
                    improved = True
                    break
        if not improved:
            break
    
    return best_h
