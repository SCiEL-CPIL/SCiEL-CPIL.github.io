import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using Gaussian Radial Basis Functions.
    
    The model is y(x) = sum_{i=0}^{9} w_i * exp(-0.5 * ((x - c_i) / sigma)^2)
    where c_i are fixed centers in [-2, 2] and sigma is a fixed width.
    This basis is well-conditioned for BFGS and can approximate a wide variety
    of smooth functions better than high-degree polynomials for non-polynomial targets.
    """
    # Fixed centers for the 10 RBFs, uniformly spaced in [-2, 2]
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Fixed width for the RBFs. 
    # Chosen to be narrow enough to capture local features 
    # but wide enough to ensure smoothness and good conditioning.
    sigma = 0.7
    
    # Compute the kernel matrix
    # x is shape (N,)
    # centers is shape (10,)
    # We need diff to be shape (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian kernel
    # Using 0.5 * (diff/sigma)^2 for standard normal PDF form
    kernel = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Linear combination of RBFs
    # kernel is (N, 10), params is (10,)
    # Result is (N,)
    return kernel @ params
