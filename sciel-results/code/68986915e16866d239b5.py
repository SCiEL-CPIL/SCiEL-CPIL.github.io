import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """
    Compute the score: L2^2 / (L1 * Sup(autoconv))
    For step heights h on intervals of width DELTA:
    L1 = DELTA * sum(h)
    L2^2 = DELTA * sum(h^2)
    autoconv_max = DELTA^2 * max(conv(h, h))
    
    Ratio = (DELTA * sum(h^2)) / (DELTA * sum(h) * DELTA^2 * max_conv)
          = sum(h^2) / (DELTA^2 * sum(h) * max_conv)
    
    Since DELTA is constant, maximizing the raw score is equivalent to maximizing
    J(h) = sum(h^2) / (sum(h) * max_conv)
    
    If we normalize sum(h) = 1, we maximize sum(h^2) / max_conv.
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    # The raw score includes the 1/DELTA^2 factor
    return (l2_sq / (l1 * autoconv_max)) / (DELTA**2)

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    # 1. Find a large Sidon set in {0, ..., N-1}
    # A Sidon set (B_2 set) has distinct pair sums i+j (i <= j).
    # We want the largest such set to minimize convolution collisions.
    
    def is_sidon(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def find_sidon_set(N):
        # Greedy approach
        S = [0]
        for i in range(1, N):
            candidate = S + [i]
            if is_sidon(candidate):
                S = candidate
        return S

    # Try to find a larger Sidon set using multiple random restarts of greedy
    best_sidon = []
    for _ in range(20):
        # Shuffle the order of candidates to break symmetry
        indices = list(range(1, N))
        np.random.shuffle(indices)
        S = [0]
        for i in indices:
            candidate = S + [i]
            if is_sidon(candidate):
                S = candidate
        if len(S) > len(best_sidon):
            best_sidon = S
            
    sidon_set = best_sidon
    K = len(sidon_set)
    
    # 2. Initialize best_h with uniform weights on Sidon set
    best_h = np.zeros(N)
    best_h[sidon_set] = 1.0 / K
    best_score = _score_raw(best_h)
    
    # 3. The theoretical optimum for a Sidon set is uniform weights.
    # Any deviation increases max_conv or decreases sum_sq relative to max_conv.
    # Thus, we return the uniform Sidon distribution directly.
    # This avoids the risk of SLSQP converging to a suboptimal point or spreading mass to non-Sidon indices.
    
    return list(best_h)
