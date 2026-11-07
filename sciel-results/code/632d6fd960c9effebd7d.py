"""Symbolic regression with RBF hidden layer."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF-based model with 10 parameters.
    
    We use 10 Gaussian radial basis functions centered at evenly spaced points
    in [0.1, 6]. The output is a weighted sum of these RBFs.
    
    y = sum_{i=0}^{9} params[i] * exp(-((x - c_i)^2) / (2 * sigma^2))
    
    where c_i are 10 evenly spaced centers in [0.1, 6] and sigma is a fixed width.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # 10 centers evenly spaced in [0.1, 6]
    centers = np.linspace(0.1, 6.0, 10)
    
    # Fixed width: choose sigma so that RBFs have reasonable overlap
    # Distance between adjacent centers: (6.0 - 0.1) / 9 ≈ 0.656
    # Use sigma = 0.8 for good coverage
    sigma = 0.8
    
    # Compute RBF values: shape (n_samples, 10)
    # x is (n_samples,), centers is (10,)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-(diff ** 2) / (2.0 * sigma ** 2))
    
    # Weighted sum: (n_samples, 10) @ (10,) -> (n_samples,)
    return rbf_values @ params
# EVOLVE-BLOCK-END
