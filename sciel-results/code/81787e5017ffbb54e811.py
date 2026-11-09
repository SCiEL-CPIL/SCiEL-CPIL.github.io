import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Radial Basis Function (RBF) model
    with five fixed centers and fixed widths.
    
    Args:
        x: Input array of shape (N,) with values in [-3, 3].
        params: Array of 10 parameters to be fitted. First 5 are weights,
                last 5 are unused (kept for interface compatibility).
        
    Returns:
        Prediction array of shape (N,).
    """
    # Five fixed centers evenly spaced across [-3, 3]
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    # Fixed widths - moderate width to capture local features
    # Width of 1.0 means Gaussian has std dev of 1.0
    widths = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    
    # Compute RBF values: exp(-0.5 * ((x - center) / width)^2)
    # x has shape (N,), centers has shape (5,)
    # diff has shape (N, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    scaled_diff = diff / widths[np.newaxis, :]
    
    # Compute squared distances
    dist2 = 0.5 * scaled_diff ** 2
    
    # Clip to avoid underflow/overflow issues
    dist2 = np.clip(dist2, 0.0, 500.0)
    
    # RBF values: exp(-dist2)
    rbf_values = np.exp(-dist2)  # shape (N, 5)
    
    # Extract weights from params (first 5 parameters)
    weights = params[:5]
    
    # Linear combination: y = sum(w_i * rbf_i(x))
    # rbf_values shape (N, 5), weights shape (5,)
    y = rbf_values @ weights
    
    # Ensure finite output
    if not np.all(np.isfinite(y)):
        return np.zeros_like(x)
        
    return y
