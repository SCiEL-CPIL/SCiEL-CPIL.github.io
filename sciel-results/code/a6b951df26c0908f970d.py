import numpy as np
from scipy.optimize import differential_evolution


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average probability measure.
    
    We design a discrete measure mu on [0, ∞) to maximize
    P(X1 + X2 + X3 < 2*X4) where X1..X4 are i.i.d. from mu.
    
    Approach: Use a small number of support points with optimized weights.
    We use differential evolution to search over support points and weights.
    The problem is scale-invariant, so we normalize the support points by their max
    inside the objective function to improve optimization stability.
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
        
        # Scale invariance: normalize support by its max
        s_max = np.max(support)
        if s_max <= 1e-9:
            return 1.0
        s = support / s_max
        
        # Compute P
        P = 0.0
        for k in range(n):
            for l in range(n):
                threshold = 2.0 * s[l] - s[k]
                inner = 0.0
                for i in range(n):
                    for j in range(n):
                        if s[i] + s[j] < threshold:
                            inner += w[i] * w[j]
                P += w[k] * w[l] * inner
        
        return -P

    best_support = None
    best_weights = None
    best_score = -1.0

    # Try n=3 and n=4
    for n in [3, 4]:
        bounds = [(0.0, 10.0)] * n + [(0.0, 1.0)] * n
        
        # Use a larger population and more iterations for better global search
        result = differential_evolution(
            neg_probability,
            bounds=bounds,
            seed=42,
            maxiter=1000,
            tol=1e-10,
            polish=True,
            workers=1,
            popsize=15
        )
        
        if np.isfinite(result.fun):
            score = -result.fun
            if score > best_score:
                # Extract best params
                support = result.x[:n].copy()
                weights = result.x[n:].copy()
                
                # Ensure non-negative and valid
                support = np.maximum(support, 0.0)
                weights = np.maximum(weights, 1e-10)
                
                w_sum = np.sum(weights)
                if w_sum <= 1e-9:
                    continue
                
                weights = weights / w_sum
                
                # Normalize support to max 1 for consistency (optional but good)
                s_max = np.max(support)
                if s_max > 1e-9:
                    support = support / s_max
                else:
                    continue
                    
                best_score = score
                best_support = support
                best_weights = weights

    # Analytical 2-point solution score
    p_opt = (1 + np.sqrt(33)) / 16.0
    analytical_score = p_opt * (1 - p_opt)**2 * (1 + 2 * p_opt)
    
    if best_support is None or best_score < analytical_score:
        support = np.array([0.0, 1.0])
        weights = np.array([1-p_opt, p_opt])
        return support.astype(np.float64), weights.astype(np.float64)

    return best_support.astype(np.float64), best_weights.astype(np.float64)
