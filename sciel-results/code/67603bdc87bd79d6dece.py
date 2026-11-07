import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers a hidden function using a linear combination of Gaussian Radial Basis Functions (RBFs).
    
    The 10 parameters are fitted by the evaluator using BFGS.
    The basis consists of 10 Gaussian bumps centered at evenly spaced points in [-3, 3].
    This local basis allows for flexible approximation of various function shapes, 
    including those with local features or non-smooth characteristics, which global 
    polynomials or low-frequency trigonometric functions may fail to capture.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Define 10 Gaussian RBF basis functions
    # Centers evenly spaced from -3 to 3
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Width (sigma) for the Gaussians. 
    # A width of ~0.6 provides good overlap for centers spaced ~0.66 apart.
    sigma = 0.6
    
    # Precompute the basis matrix
    # b[i] is the value of the i-th RBF for all x
    # We can compute this efficiently
    
    # Initialize output
    result = np.zeros_like(x)
    
    for i in range(10):
        # Calculate Gaussian: exp(-((x - center) / sigma)^2)
        # Using squared distance for stability
        diff = x - centers[i]
        b_i = np.exp(-(diff * diff) / (sigma * sigma))
        
        # Add contribution
        result += params[i] * b_i
        
    return result
