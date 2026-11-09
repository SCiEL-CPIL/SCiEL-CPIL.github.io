import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Radial Basis Function (RBF) network.
    
    This strategy uses 5 fixed, equally spaced Gaussian centers across the domain [-3, 3].
    The parameters control the weights of these 5 kernels, a global offset, and a scaling
    factor for the kernel width. This localized basis is structurally different from the
    global polynomial basis of the previous approach, potentially allowing better
    approximation of functions with sharp features or non-polynomial behavior.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    p = params[:10]
    
    # Ensure finite output
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
    
    # 5 fixed centers equally spaced in [-3, 3]
    # Centers: -3, -1.5, 0, 1.5, 3
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    # Parameters:
    # p[0:5] -> weights for the 5 kernels
    # p[5]   -> global offset (bias)
    # p[6]   -> width scaling factor (sigma scale)
    # p[7:10] -> unused (spare)
    
    weights = p[0:5]
    offset = p[5]
    
    # Base sigma for kernels. Domain is 6 units wide, 5 centers spaced by 1.5.
    # A reasonable base sigma might be around 1.0.
    # We use p[6] to scale this. To ensure stability and avoid division by zero or extreme values,
    # we constrain the effective sigma.
    base_sigma = 1.0
    # Softplus or sigmoid-like scaling to keep sigma positive and reasonable?
    # The evaluator uses BFGS on raw params. It's safer to map p[6] to a positive range.
    # Let sigma = base_sigma * (1 + np.exp(p[6]) / (1 + np.exp(-p[6])))? No, simpler:
    # Let's use sigma = base_sigma * (1.0 + np.abs(p[6])) or similar.
    # Actually, allowing the width to be optimized is good.
    # Let's define sigma = 1.0 / (1.0 + np.exp(-p[6])) * 3.0 + 0.1
    # This maps p[6] to sigma in [0.1, 3.1]. This is robust.
    sigma = 0.1 + 3.0 / (1.0 + np.exp(-p[6]))
    
    # Compute Gaussian kernels
    # x shape (n,), centers shape (5,)
    # diff shape (n, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # K = exp(-0.5 * (diff / sigma)^2)
    K = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Prediction
    y_pred = K @ weights + offset
    
    return y_pred
