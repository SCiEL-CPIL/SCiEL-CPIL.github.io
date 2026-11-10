import numpy as np
from scipy.optimize import differential_evolution


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average probability measure.
    
    We design a discrete measure mu on [0, ∞) to maximize
    P(X1 + X2 + X3 < 2*X4) where X1..X4 are i.i.d. from mu.
    
    Approach: Use a small number of support points with optimized weights.
    We use differential evolution to search over support points and weights.
    """
    def neg_probability(params):
        """Objective function to minimize (negative probability)."""
        n = len(params) // 2
        support = params[:n]
        weights = params[n:]
        
        # Ensure valid values
        if np.any(support < -1e-9) or np.any(weights < -1e-9):
            return 1.0  # Large penalty
        
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 1.0
        
        # Normalize weights
        w = weights / w_sum
        
        # Compute P = sum over k, l of w_k * w_l * (sum of w_i * w_j over pairs (i, j) with s_i + s_j < 2*s_l - s_k)
        # For n points, this is O(n^4) naively, but n is small here
        P = 0.0
        for k in range(n):
            for l in range(n):
                threshold = 2.0 * support[l] - support[k]
                inner = 0.0
                for i in range(n):
                    for j in range(n):
                        if support[i] + support[j] < threshold:
                            inner += w[i] * w[j]
                P += w[k] * w[l] * inner
        
        return -P
    
    # Try with n=3 support points
    # Support points in [0, 5], weights positive
    n = 3
    bounds = [(0.0, 5.0)] * n + [(1e-4, 1.0)] * n
    
    result = differential_evolution(
        neg_probability,
        bounds=bounds,
        seed=42,
        maxiter=500,
        tol=1e-8,
        polish=True,
        workers=1
    )
    
    support = result.x[:n].copy()
    weights = result.x[n:].copy()
    
    # Ensure non-negative
    support = np.maximum(support, 0.0)
    weights = np.maximum(weights, 1e-10)
    
    # Normalize weights
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        weights = np.ones(n) / n
    else:
        weights = weights / w_sum
    
    return support.astype(np.float64), weights.astype(np.float64)
