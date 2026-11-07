import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Gaussian Radial Basis Function (RBF) network.
    
    The model uses 10 Gaussian RBFs with fixed centers uniformly distributed 
    over the input domain [-2, 2]. The 10 parameters are the weights (amplitudes) 
    of these basis functions.
    
    y(x) = sum_{j=0}^{9} w_j * exp(-((x - c_j)^2) / (2 * sigma^2))
    
    This approach is chosen because:
    1. It avoids the numerical instability of high-degree polynomial bases 
       (like Chebyshev degree 9) which can cause BFGS to diverge.
    2. RBFs provide local flexibility, allowing the model to fit non-linear 
       structures without global oscillations.
    3. The design matrix is typically well-conditioned for uniform centers 
       and moderate widths, aiding BFGS convergence.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure x is 1D for consistent broadcasting, though it should be 1D already
    x_flat = x.flatten()
    
    # Define 10 fixed centers uniformly spaced in [-2, 2]
    # j = 0 -> -2.0, j = 9 -> 2.0
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Define a fixed width (sigma) for the Gaussians.
    # A width of ~0.8 ensures good overlap between adjacent centers,
    # allowing the linear combination to form smooth curves.
    sigma = 0.8
    sigma_sq_2 = 2.0 * sigma * sigma
    
    # Compute the RBF basis matrix: shape (N_points, 10)
    # diff[i, j] = x_flat[i] - centers[j]
    diff = x_flat[:, np.newaxis] - centers[np.newaxis, :]
    
    # Compute exp(-diff^2 / (2*sigma^2))
    # We clip the exponent to avoid underflow/overflow issues, though for 
    # x in [-2,2] and centers in [-2,2], the max distance is 4, 
    # -16 / (2*0.64) = -12.5, which is safe for float64.
    basis = np.exp(-(diff * diff) / sigma_sq_2)
    
    # Compute prediction: sum of weights * basis
    # params shape (10,), basis shape (N, 10)
    # result shape (N,)
    result = basis @ params
    
    # Reshape back to original x shape
    result = result.reshape(x.shape)
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
        
    return result
