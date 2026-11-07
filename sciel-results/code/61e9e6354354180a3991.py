import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression model using 3 Gaussian Radial Basis Functions and a bias term.
    This structure allows for local feature adaptation, which is effective for
    non-linear targets with localized variations.
    
    Parameters:
    params[0]: Bias
    params[1]: Weight for RBF 1
    params[2]: Center for RBF 1
    params[3]: Sigma (width) for RBF 1
    params[4]: Weight for RBF 2
    params[5]: Center for RBF 2
    params[6]: Sigma (width) for RBF 2
    params[7]: Weight for RBF 3
    params[8]: Center for RBF 3
    params[9]: Sigma (width) for RBF 3
    """
    b = params[0]
    
    # RBF 1
    w1 = params[1]
    c1 = params[2]
    # Use absolute value plus a small epsilon to ensure positive sigma
    # This avoids division by zero and negative sigma issues
    s1 = np.abs(params[3]) + 1e-3
    
    # RBF 2
    w2 = params[4]
    c2 = params[5]
    s2 = np.abs(params[6]) + 1e-3
    
    # RBF 3
    w3 = params[7]
    c3 = params[8]
    s3 = np.abs(params[9]) + 1e-3
    
    # Compute RBF terms
    # The exponent is always non-positive, so no overflow risk
    rbf1 = w1 * np.exp(-(x - c1)**2 / (s1**2))
    rbf2 = w2 * np.exp(-(x - c2)**2 / (s2**2))
    rbf3 = w3 * np.exp(-(x - c3)**2 / (s3**2))
    
    return b + rbf1 + rbf2 + rbf3
