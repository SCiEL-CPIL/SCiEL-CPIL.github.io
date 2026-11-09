import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Scale input to [-1, 1]
    x_scaled = x / 3.0
    
    # 5 fixed centers evenly spaced in [-1, 1]
    centers = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    weights = p[0:5]
    
    # Widths using softplus for better optimization landscape
    # Clip to prevent extreme values
    raw_widths = np.clip(p[5:10], -10, 10)
    sigma = np.log1p(np.exp(raw_widths)) # softplus
    
    # Compute RBF values
    diff = x_scaled[:, np.newaxis] - centers[np.newaxis, :]
    scaled_diff = diff / sigma[np.newaxis, :]
    
    # Gaussian RBF
    rbf_values = np.exp(-0.5 * scaled_diff ** 2)
    
    # Weighted sum
    prediction = rbf_values @ weights
    
    return prediction
