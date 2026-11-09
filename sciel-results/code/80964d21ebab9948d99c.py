import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # 10 centers uniformly distributed in [-2, 2]
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Fixed width for Gaussians
    # sigma = 0.5 -> gamma = 1.0 / (0.5**2) = 4.0
    gamma = 4.0
    
    # Compute basis functions: exp(-gamma * (x - c)^2)
    # x is (N, 1) or (N,), centers is (10,)
    # We need to broadcast correctly.
    # x[:, np.newaxis] - centers[np.newaxis, :] gives (N, 10)
    # If x is 1D, x[:, np.newaxis] works.
    
    x_expanded = x[:, np.newaxis]
    dist_sq = (x_expanded - centers[np.newaxis, :]) ** 2
    basis = np.exp(-gamma * dist_sq)
    
    # params is (10,)
    # result is (N, 1) or (N,)
    # basis @ params gives (N, 1) if basis is (N, 10) and params is (10,)
    # We want to return array with one prediction per point.
    result = basis @ params
    
    # Ensure output shape matches input shape (1D array)
    return result.flatten()
