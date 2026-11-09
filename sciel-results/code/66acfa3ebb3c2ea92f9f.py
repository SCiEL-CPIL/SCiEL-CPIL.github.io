"""Symbolic regression with 2 inputs: Hybrid RBF + Chebyshev basis for local-global representation."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 145.0  # safety margin before the 150s kill


def _features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Return a (n, 10) feature matrix combining global Chebyshev polynomials and local RBF terms.
    
    Strategy:
    - 5 global Chebyshev tensor-product features (degree <= 2) to capture smooth global trends.
    - 5 local RBF (Gaussian) features centered at key points to capture local structure.
    
    This hybrid approach allows the model to represent both smooth polynomial-like functions
    and functions with localized features (e.g., bumps, sharp transitions) that pure polynomials
    struggle with.
    """
    n = x.shape[0]

    # --- Global Chebyshev features (degree <= 2) ---
    # Chebyshev polynomials: T0=1, T1=x, T2=2x^2-1
    Tx0 = np.ones(n)
    Tx1 = x
    Tx2 = 2.0 * x * x - 1.0

    Ty0 = np.ones(n)
    Ty1 = y
    Ty2 = 2.0 * y * y - 1.0

    # Scale non-constant terms to have unit variance under uniform [-1,1]
    S = np.sqrt(2.0)

    # 5 global features:
    f0 = Tx0 * Ty0  # 1 (constant)
    f1 = S * Tx1 * Ty0  # sqrt(2) * x
    f2 = S * Tx0 * Ty1  # sqrt(2) * y
    f3 = S * Tx2 * Ty0  # sqrt(2) * (2x^2 - 1)
    f4 = S * S * Tx1 * Ty1  # 2 * x * y

    # --- Local RBF features ---
    # 5 Gaussian RBFs centered at: (0,0), (0.5,0.5), (-0.5,-0.5), (0.5,-0.5), (-0.5,0.5)
    # Width sigma = 0.4 for moderate locality
    sigma = 0.4
    centers = [
        (0.0, 0.0),
        (0.5, 0.5),
        (-0.5, -0.5),
        (0.5, -0.5),
        (-0.5, 0.5),
    ]

    rbf_features = []
    for cx, cy in centers:
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        rbf = np.exp(-dist_sq / (2.0 * sigma * sigma))
        rbf_features.append(rbf)

    # Combine: 5 global + 5 local = 10 features
    F = np.column_stack([f0, f1, f2, f3, f4] + rbf_features)
    return F


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 parameters.
    Uses a hybrid basis of global Chebyshev polynomials and local RBF terms.
    """
    # Guard against excessive runtime
    if time.time() > _DEADLINE:
        F = _features(x, y)
        return F @ params

    F = _features(x, y)
    pred = F @ params

    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        mask = np.isfinite(pred)
        pred[~mask] = 0.0

    return pred
# EVOLVE-BLOCK-END
