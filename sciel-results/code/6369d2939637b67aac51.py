import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a sequence of 50 nonnegative heights to maximize
    R = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    using numerical search over various families.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    def compute_ratio(h):
        h = np.array(h, dtype=float)
        if np.all(h == 0):
            return 0.0
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        if l1 == 0:
            return 0.0
        # Linear autoconvolution
        conv = np.convolve(h, h, mode='full')
        conv_inf = np.max(conv)
        if conv_inf == 0:
            return 0.0
        return l2_sq / (l1 * conv_inf)
    
    best_h = [0.0] * N
    best_score = 0.0
    
    # Try single point
    h_test = [0.0] * N
    h_test[0] = 1.0
    s = compute_ratio(h_test)
    if s > best_score:
        best_score = s
        best_h = h_test[:]
    
    # Try 2-point clusters with various ratios
    for a in np.linspace(0.01, 0.99, 100):
        if time.time() > deadline:
            break
        h_test = [0.0] * N
        h_test[0] = a
        h_test[1] = 1.0 - a
        s = compute_ratio(h_test)
        if s > best_score:
            best_score = s
            best_h = h_test[:]
    
    # Try 3-point clusters
    for a in np.linspace(0.1, 0.8, 20):
        for b in np.linspace(0.1, 0.8, 20):
            if time.time() > deadline:
                break
            c = 1.0 - a - b
            if c < 0.01:
                continue
            h_test = [0.0] * N
            h_test[0] = a
            h_test[1] = b
            h_test[2] = c
            s = compute_ratio(h_test)
            if s > best_score:
                best_score = s
                best_h = h_test[:]
    
    # Try uniform over k points
    for k in range(1, 11):
        if time.time() > deadline:
            break
        h_test = [0.0] * N
        for i in range(k):
            h_test[i] = 1.0 / k
        s = compute_ratio(h_test)
        if s > best_score:
            best_score = s
            best_h = h_test[:]
    
    # Try exponential decay
    for alpha in np.linspace(0.1, 2.0, 30):
        if time.time() > deadline:
            break
        h_raw = np.exp(-alpha * np.arange(N))
        h_norm = h_raw / np.sum(h_raw)
        s = compute_ratio(h_norm)
        if s > best_score:
            best_score = s
            best_h = list(h_norm)
    
    # Try Gaussian-like
    for sigma in np.linspace(0.5, 5.0, 20):
        if time.time() > deadline:
            break
        x = np.arange(N) - N/2
        h_raw = np.exp(-x**2 / (2*sigma**2))
        h_norm = h_raw / np.sum(h_raw)
        s = compute_ratio(h_norm)
        if s > best_score:
            best_score = s
            best_h = list(h_norm)
    
    # Local search: perturb best solution
    if time.time() < deadline - 5:
        h_best = np.array(best_h)
        for _ in range(100):
            if time.time() > deadline:
                break
            # Random perturbation
            h_pert = h_best.copy()
            idx = np.random.randint(0, N)
            delta = np.random.uniform(-0.05, 0.05)
            h_pert[idx] = max(0, h_pert[idx] + delta)
            # Renormalize to L1=1
            if np.sum(h_pert) > 0:
                h_pert = h_pert / np.sum(h_pert)
            s = compute_ratio(h_pert)
            if s > best_score:
                best_score = s
                best_h = list(h_pert)
    
    return best_h
# EVOLVE-BLOCK-END
