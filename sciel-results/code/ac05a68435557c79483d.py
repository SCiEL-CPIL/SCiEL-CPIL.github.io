# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] exactly."""
    n = len(support)
    # Normalize weights
    w = weights / weights.sum()
    s = support
    
    # Build pairwise sums s_i + s_j
    # P = sum_{k,l} w_k * w_l * (sum_{i,j: s_i+s_j < 2*s_l - s_k} w_i * w_j)
    # For efficiency, precompute for each (k,l) the threshold and use cumulative sums
    
    # For each pair (i,j), compute s_i + s_j
    # Then for each (k,l), we need sum of w_i*w_j where s_i+s_j < 2*s_l - s_k
    
    # This is O(n^4) naively. Let's use a smarter approach.
    # For fixed k and l, threshold T = 2*s_l - s_k
    # We need sum_{i,j: s_i+s_j < T} w_i * w_j
    
    # Precompute all pairwise sums and their weights
    # s_i + s_j for all i,j
    ss = s[:, None] + s[None, :]  # n x n
    ww = w[:, None] * w[None, :]  # n x n
    
    # For each (k,l), threshold T_kl = 2*s_l - s_k
    # P = sum_{k,l} w_k * w_l * (sum_{i,j: ss[i,j] < T_kl} ww[i,j])
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            T = 2.0 * s[l] - s[k]
            if T <= 0:
                continue
            # Sum ww where ss < T
            mask = ss < T
            P += w[k] * w[l] * ww[mask].sum()
    
    return P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Returns a discrete measure maximizing P[X1+X2+X3 < 2*X4].
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    # Start with the 2-point optimal
    p2 = 1.0 / np.sqrt(10.0)
    support2 = np.array([0.0, 1.0], dtype=np.float64)
    weights2 = np.array([1.0 - p2, p2], dtype=np.float64)
    P_best = _compute_P(support2, weights2)
    support_best = support2.copy()
    weights_best = weights2.copy()
    
    # Try 3-point distributions: {0, a, b} with a < b
    # By scaling invariance, set b=1, vary a and the weights
    # Actually, let's try a few structured distributions
    
    # Try uniform on {0, 1, ..., n-1} for various n
    for n in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32]:
        if time.time() > deadline:
            break
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
        P = _compute_P(support, weights)
        if P > P_best:
            P_best = P
            support_best = support.copy()
            weights_best = weights.copy()
    
    # Try geometric-like: support = {0, 1, 2, ..., n-1} with geometric weights
    for n in [4, 6, 8, 10, 12, 16, 20]:
        if time.time() > deadline:
            break
        for r in [0.5, 0.6, 0.7, 0.8, 0.9]:
            support = np.arange(n, dtype=np.float64)
            weights = np.array([r**i for i in range(n)], dtype=np.float64)
            P = _compute_P(support, weights)
            if P > P_best:
                P_best = P
                support_best = support.copy()
                weights_best = weights.copy()
    
    # Try 3-point: {0, a, 1} with various a and weight splits
    for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        if time.time() > deadline:
            break
        for w1 in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for w2 in [0.05, 0.1, 0.2, 0.3]:
                w0 = 1.0 - w1 - w2
                if w0 < 1e-9:
                    continue
                support = np.array([0.0, a, 1.0], dtype=np.float64)
                weights = np.array([w0, w1, w2], dtype=np.float64)
                P = _compute_P(support, weights)
                if P > P_best:
                    P_best = P
                    support_best = support.copy()
                    weights_best = weights.copy()
    
    return support_best, weights_best


# EVOLVE-BLOCK-END
