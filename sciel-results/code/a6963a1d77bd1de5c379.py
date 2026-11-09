"""Symbolic regression with 1 input(s) using a hybrid Chebyshev-RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the unknown function using a linear combination of a hybrid basis:
    - 5 Chebyshev polynomials (mapped to [-1, 1]) for global smooth structure.
    - 5 Gaussian RBFs (centers at 1, 2, 3, 4, 5) for local flexibility.
    
    This basis is chosen to cover a wide range of common functional forms:
    Chebyshev terms capture global trends and oscillatory behavior efficiently,
    while RBFs capture local bumps, peaks, and sharp features.
    """
    x = np.asarray(x, dtype=np.float64)
    
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
        
    n_points = x.shape[0]
    if n_points == 0:
        return np.array([])
        
    # Build the basis matrix
    # Shape: (n_points, 10)
    basis = np.zeros((n_points, 10), dtype=np.float64)
    
    # Part 1: Chebyshev polynomials (5 terms: T_0 to T_4)
    # Map x from [0.1, 6] to t in [-1, 1]
    # t = (2*x - (min+max)) / (max-min)
    # min=0.1, max=6.0
    t = (2.0 * x - 6.1) / 5.9
    t = np.clip(t, -1.0, 1.0)
    
    # T_0(t) = 1
    basis[:, 0] = 1.0
    
    # T_1(t) = t
    basis[:, 1] = t
    
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    for k in range(2, 5):
        basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
    
    # Part 2: Gaussian RBFs (5 terms)
    # Centers evenly spaced: 1.0, 2.0, 3.0, 4.0, 5.0
    # Bandwidth sigma = 0.8
    centers = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sigma = 0.8
    
    # Compute RBFs: exp(-(x - c)^2 / (2 * sigma^2))
    # x is (N,), centers is (5,)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_terms = np.exp(-(diff ** 2) / (2.0 * sigma ** 2))
    
    basis[:, 5:10] = rbf_terms
    
    # Linear combination
    # params is shape (10,)
    pred = basis @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        return np.zeros_like(x)
        
    return pred
# EVOLVE-BLOCK-END
