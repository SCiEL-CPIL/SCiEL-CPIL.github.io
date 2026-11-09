# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] exactly for discrete measure."""
    n = len(support)
    w = weights / weights.sum()
    # Pairwise sums: sum_{i,j} w_i w_j * 1[s_i + s_j < 2*s_l - s_k]
    # For each (k, l), compute sum of w_i*w_j over pairs with s_i+s_j < 2*s_l - s_k
    # Use vectorized approach
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * support[l] - support[k]
            # Need sum of w_i * w_j for all (i,j) with support[i] + support[j] < threshold
            # Build pairwise sums matrix
            pair_sums = support[:, None] + support[None, :]
            mask = pair_sums < threshold
            inner = np.sum(w[:, None] * w[None, :] * mask)
            P += w[k] * w[l] * inner
    return P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game using a Pareto-like heavy-tail structural family.
    
    We use a discrete Pareto distribution: support points at geometrically spaced
    locations with weights following a power-law decay. This creates a heavy tail
    where large X4 values make 2*X4 frequently exceed X1+X2+X3.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    deadline = time.time() + 580  # safety margin before 630s limit
    
    best_score = -1.0
    best_support = None
    best_weights = None
    
    # Start with baseline for reference
    n_base = 32
    base_support = np.arange(n_base, dtype=np.float64)
    base_weights = np.ones(n_base, dtype=np.float64) / n_base
    base_P = _compute_P(base_support, base_weights)
    if base_P > best_score:
        best_score = base_P
        best_support = base_support.copy()
        best_weights = base_weights.copy()
    
    # Explore Pareto-like discrete distributions
    # Support: geometrically spaced points [a * r^i for i in range(n)]
    # Weights: power-law [c * alpha^i for i in range(n)]
    # Parameters: a (base scale), r (geometric ratio), alpha (weight decay), n (num points)
    
    # Grid search over key parameters
    n_values = [16, 24, 32, 48, 64]
    r_values = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    alpha_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for n in n_values:
        if time.time() > deadline:
            break
        for r in r_values:
            if time.time() > deadline:
                break
            for alpha in alpha_values:
                if time.time() > deadline:
                    break
                support = np.array([r ** i for i in range(n)], dtype=np.float64)
                weights = np.array([alpha ** i for i in range(n)], dtype=np.float64)
                # Ensure non-negative and valid
                if np.any(support < -1e-9) or np.any(weights < -1e-9):
                    continue
                if weights.sum() <= 1e-9:
                    continue
                try:
                    P = _compute_P(support, weights)
                    if P > best_score:
                        best_score = P
                        best_support = support.copy()
                        best_weights = weights.copy()
                except Exception:
                    continue
    
    # Also try two-point and three-point optimized solutions
    # Two-point: {0, M} with weights {p, 1-p}
    # Optimal p from analysis: p ≈ 0.5785, any M > 0
    p_opt = (15 - np.sqrt(33)) / 16
    for M in [10, 100, 1000, 10000]:
        support = np.array([0.0, M], dtype=np.float64)
        weights = np.array([p_opt, 1 - p_opt], dtype=np.float64)
        try:
            P = _compute_P(support, weights)
            if P > best_score:
                best_score = P
                best_support = support.copy()
                best_weights = weights.copy()
        except Exception:
            pass
    
    # Three-point: {0, 1, M} with weights {p0, p1, p2}
    # Grid search over p0, p1 (p2 = 1 - p0 - p1)
    for M in [10, 100, 1000]:
        if time.time() > deadline:
            break
        for p0 in np.linspace(0.1, 0.8, 15):
            if time.time() > deadline:
                break
            for p1 in np.linspace(0.05, 0.7, 15):
                p2 = 1.0 - p0 - p1
                if p2 < 0.01:
                    continue
                support = np.array([0.0, 1.0, M], dtype=np.float64)
                weights = np.array([p0, p1, p2], dtype=np.float64)
                try:
                    P = _compute_P(support, weights)
                    if P > best_score:
                        best_score = P
                        best_support = support.copy()
                        best_weights = weights.copy()
                except Exception:
                    continue
    
    # Final validation
    if best_support is None or best_weights is None:
        # Fallback to baseline
        return base_support, base_weights
    
    # Ensure validity
    best_support = np.asarray(best_support, dtype=np.float64).flatten()
    best_weights = np.asarray(best_weights, dtype=np.float64).flatten()
    
    # Clamp tiny negatives
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    # Ensure sum of weights > 1e-9
    if best_weights.sum() <= 1e-9:
        return base_support, base_weights
    
    return best_support, best_weights


# EVOLVE-BLOCK-END
