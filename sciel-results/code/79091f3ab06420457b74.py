# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] exactly in float64."""
    n = len(support)
    # Pairwise sums: S[i,j] = support[i] + support[j]
    S = support[:, None] + support[None, :]
    # For each (k, l) pair representing (X4, threshold), compute sum over (i,j) with S[i,j] < 2*support[l] - support[k]
    # P = sum_k sum_l w_k w_l * (sum_{i,j: S[i,j] < 2*s_l - s_k} w_i w_j)
    # Vectorized: for each l, threshold_t = 2*s_l - s_k for each k
    # Let's compute: for each l, we have thresholds T_k = 2*s_l - s_k for k=0..n-1
    # Then for each k, we need sum of w_i*w_j where S[i,j] < T_k
    # Precompute all pairwise weighted sums? That's O(n^2) per (k,l) which is O(n^4) total.
    
    # Better: for each l, compute the matrix of indicators S < (2*s_l - s_k) for all k
    # Actually, let's do: for each l, thresholds = 2*support[l] - support (vector of length n)
    # For each k, we need cumulative sum of w_i*w_j over S[i,j] < thresholds[k]
    
    # Precompute W = w_i * w_j (outer product)
    W = np.outer(weights, weights)
    # Sort S and W together? Or use searchsorted.
    # Flatten S and W
    S_flat = S.ravel()
    W_flat = W.ravel()
    # Sort by S_flat
    order = np.argsort(S_flat)
    S_sorted = S_flat[order]
    W_sorted = W_flat[order]
    # Cumulative sum of W_sorted
    W_cumsum = np.cumsum(W_sorted)
    
    P_total = 0.0
    for l in range(n):
        # thresholds[k] = 2*support[l] - support[k]
        thresholds = 2.0 * support[l] - support
        # For each k, find count of S_sorted < thresholds[k]
        # Using searchsorted
        counts = np.searchsorted(S_sorted, thresholds, side='left')
        # Sum of W for indices < counts[k]
        # W_cumsum[counts[k]-1] gives sum up to index counts[k]-1
        sums = np.zeros(n)
        mask = counts > 0
        sums[mask] = W_cumsum[counts[mask] - 1]
        # P contribution: sum_k w_k * w_l * sums[k]
        P_total += weights[l] * np.dot(weights, sums)
    
    return P_total


def _try_distribution(support, weights):
    """Try a distribution and return P."""
    # Validate
    if len(support) != len(weights):
        return -1.0
    if len(support) < 1:
        return -1.0
    if np.any(~np.isfinite(support)) or np.any(~np.isfinite(weights)):
        return -1.0
    if np.any(support < -1e-9) or np.any(weights < -1e-9):
        return -1.0
    if np.sum(weights) <= 1e-9:
        return -1.0
    return _compute_P(support, weights)


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4].
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600s wall-clock limit
    
    # Start with best known: two-point {0, L} with optimal p
    # p* = (1 + sqrt(33))/16 ≈ 0.42157
    p_opt = (1.0 + np.sqrt(33.0)) / 16.0
    best_support = np.array([0.0, 1.0], dtype=np.float64)
    best_weights = np.array([1.0 - p_opt, p_opt], dtype=np.float64)
    best_P = _try_distribution(best_support, best_weights)
    
    # Try various strategies
    strategies = []
    
    # Strategy 1: Two-point {0, L} for various L (should be same P, but verify)
    for L in [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]:
        s = np.array([0.0, L], dtype=np.float64)
        w = np.array([1.0 - p_opt, p_opt], dtype=np.float64)
        P = _try_distribution(s, w)
        if P > best_P:
            best_P = P
            best_support = s.copy()
            best_weights = w.copy()
    
    # Strategy 2: Three-point {0, a, b} with grid search
    # Try {0, 1, L} for various L and weight distributions
    for L in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        # Try different weight splits: (w0, w1, wL)
        for w1 in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for wL in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]:
                w0 = 1.0 - w1 - wL
                if w0 < 0:
                    continue
                s = np.array([0.0, 1.0, L], dtype=np.float64)
                w = np.array([w0, w1, wL], dtype=np.float64)
                P = _try_distribution(s, w)
                if P > best_P:
                    best_P = P
                    best_support = s.copy()
                    best_weights = w.copy()
                if time.time() > deadline - 30:
                    return best_support, best_weights
    
    # Strategy 3: More points - geometric-like distribution
    # Try {0, 1, 2, ..., M} with geometric weights
    for M in [3, 4, 5, 6, 8, 10, 15]:
        for ratio in [0.3, 0.4, 0.5, 0.6, 0.7]:
            # Geometric weights: w_i ∝ ratio^i
            raw_w = np.array([ratio ** i for i in range(M + 1)], dtype=np.float64)
            w = raw_w / np.sum(raw_w)
            s = np.arange(M + 1, dtype=np.float64)
            P = _try_distribution(s, w)
            if P > best_P:
                best_P = P
                best_support = s.copy()
                best_weights = w.copy()
            if time.time() > deadline - 30:
                return best_support, best_weights
    
    # Strategy 4: Concentrated at 0 with sparse large values
    # {0, L1, L2, ..., Lk} with most mass at 0
    for n_large in [1, 2, 3, 4, 5]:
        for spacing in [1.0, 2.0, 5.0, 10.0]:
            large_vals = np.array([spacing * (i + 1) for i in range(n_large)], dtype=np.float64)
            s = np.concatenate([np.array([0.0]), large_vals])
            # Try different mass allocations
            for mass_at_zero in [0.5, 0.6, 0.7, 0.8, 0.9]:
                remaining = 1.0 - mass_at_zero
                # Equal split among large values
                w_large = remaining / n_large
                w = np.concatenate([np.array([mass_at_zero]), np.full(n_large, w_large)])
                P = _try_distribution(s, w)
                if P > best_P:
                    best_P = P
                    best_support = s.copy()
                    best_weights = w.copy()
            if time.time() > deadline - 30:
                return best_support, best_weights
    
    # Strategy 5: Fine-tune the two-point solution with more structure
    # Try {0, a, 2a, 3a} with optimized weights
    for a in [0.5, 1.0, 2.0]:
        for w1 in np.linspace(0.1, 0.5, 10):
            for w2 in np.linspace(0.05, 0.3, 10):
                for w3 in np.linspace(0.02, 0.2, 10):
                    w0 = 1.0 - w1 - w2 - w3
                    if w0 < 0:
                        continue
                    s = np.array([0.0, a, 2*a, 3*a], dtype=np.float64)
                    w = np.array([w0, w1, w2, w3], dtype=np.float64)
                    P = _try_distribution(s, w)
                    if P > best_P:
                        best_P = P
                        best_support = s.copy()
                        best_weights = w.copy()
                    if time.time() > deadline - 30:
                        return best_support, best_weights
    
    # Final: try a broader grid on 4-point {0, 1, 2, L}
    for L in [3.0, 5.0, 10.0, 20.0]:
        for w1 in [0.1, 0.15, 0.2, 0.25, 0.3]:
            for w2 in [0.05, 0.1, 0.15, 0.2]:
                w3 = 0.1  # weight at L
                w0 = 1.0 - w1 - w2 - w3
                if w0 < 0:
                    continue
                s = np.array([0.0, 1.0, 2.0, L], dtype=np.float64)
                w = np.array([w0, w1, w2, w3], dtype=np.float64)
                P = _try_distribution(s, w)
                if P > best_P:
                    best_P = P
                    best_support = s.copy()
                    best_weights = w.copy()
                if time.time() > deadline - 30:
                    return best_support, best_weights
    
    # Ensure valid return
    if np.any(best_support < -1e-9) or np.any(best_weights < -1e-9) or np.sum(best_weights) <= 1e-9:
        # Fallback to two-point
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([1.0 - p_opt, p_opt], dtype=np.float64)
    
    return best_support, best_weights

# EVOLVE-BLOCK-END
