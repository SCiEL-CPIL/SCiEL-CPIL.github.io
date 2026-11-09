import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Use Radial Basis Functions (Gaussian) with fixed centers
    # This creates a linear model in 'params', which BFGS can solve robustly.
    # Fixed centers avoid the non-convex optimization of center positions.
    
    n = 10
    # Centers evenly spaced across the input domain [-3, 3]
    centers = np.linspace(-3.0, 3.0, n)
    
    # Standard deviation of the Gaussian kernels
    # Tuned for the domain size (6.0) and number of points (10)
    # sigma = 1.0 provides good overlap between neighboring kernels
    sigma = 1.0
    
    # Calculate the RBF values for each point x and each center
    # x: shape (N,)
    # centers: shape (n,)
    # diff: shape (N, n)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian kernel: exp(-0.5 * (diff / sigma)^2)
    # Using 0.5 * (x/sigma)^2 is equivalent to -x^2 / (2 * sigma^2)
    rbf_values = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Linear combination of RBFs with parameters as weights
    # rbf_values: (N, n), params: (n,) -> y: (N,)
    y = rbf_values @ params
    
    return y
