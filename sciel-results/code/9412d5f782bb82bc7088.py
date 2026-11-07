import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model using a hybrid basis combining global polynomial trends and local
    Gaussian RBF kernels for adaptive smoothing.
    
    y(x) = p_0 + p_1*x + sum_{i=0}^{7} p_{i+2} * exp(-(x - c_i)^2 / (2 * sigma_i^2))
    
    Parameters (length 10):
    - p_0: constant offset
    - p_1: linear trend coefficient
    - p_2 ... p_9: weights for 8 Gaussian RBF kernels
    
    Centers c_i are evenly spaced in [0.1, 6].
    Widths sigma_i are fixed but slightly varied to provide multi-scale local flexibility.
    
    This structure combines the global trend capture of polynomials (inspired by the
    linear trend finding) with the local interpolation capability of RBFs (current approach),
    offering a more flexible model than pure RBFs or pure polynomials.
    """
    x = np.asarray(x, dtype=float)
    x_flat = x.ravel()
    n_points = x_flat.shape[0]
    
    # Unpack parameters
    p_const = params[0]
    p_linear = params[1]
    rbf_weights = params[2:10]  # 8 RBF weights
    
    # Define RBF centers evenly spaced in the domain [0.1, 6]
    n_rbf = 8
    x_min = 0.1
    x_max = 6.0
    centers = np.linspace(x_min, x_max, n_rbf)
    
    # Define widths (sigma) for the Gaussian kernels
    # Use a base sigma of 0.5 with slight variation for multi-scale flexibility
    base_sigma = 0.5
    # Vary sigma slightly: alternate between base_sigma and base_sigma * 1.2
    sigmas = np.array([base_sigma * (1.0 + 0.2 * (i % 2)) for i in range(n_rbf)])
    sigma2 = 2.0 * sigmas ** 2
    
    # Compute the RBF values
    # diff shape: (N, 8)
    diff = x_flat[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian kernel: exp(-(x - c)^2 / (2 * sigma^2))
    # rbf shape: (N, 8)
    rbf = np.exp(-(diff ** 2) / sigma2)
    
    # Linear combination: constant + linear + RBF terms
    y = p_const + p_linear * x_flat + np.dot(rbf, rbf_weights)
    
    # Reshape to match input shape
    y = y.reshape(x.shape)
    
    return y
