import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian Radial Basis Function (RBF) model with 10 parameters.
    
    The model is:
    y = sum_{i=0}^{9} p[i] * exp(-(x - c[i])^2 / (2 * sigma^2))
    
    Centers c[i] are fixed and evenly spaced across the domain [0.1, 6].
    Sigma is fixed to provide sufficient overlap for smooth approximation.
    This is a linear model in parameters, ensuring BFGS converges reliably.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure params are finite
    if not np.all(np.isfinite(p)):
        p = np.where(np.isfinite(p), p, 0.0)
        
    # Domain: [0.1, 6.0]
    x_min = 0.1
    x_max = 6.0
    n_centers = 10
    
    # Evenly spaced centers
    # c_i = x_min + i * (x_max - x_min) / (n_centers - 1)
    centers = np.linspace(x_min, x_max, n_centers)
    
    # Width parameter
    # A width of ~0.5-0.7 usually works well for 10 points on this scale
    sigma = 0.6
    
    # Compute basis functions
    # x is shape (N,), centers is shape (10,)
    # We want result shape (N,)
    # diff is shape (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    exponent = -0.5 * (diff / sigma) ** 2
    basis = np.exp(exponent)
    
    # Linear combination
    # basis is (N, 10), p is (10,)
    prediction = basis @ p
    
    # Check for non-finite values
    if not np.all(np.isfinite(prediction)):
        prediction = np.where(np.isfinite(prediction), prediction, 0.0)
        
    return prediction
