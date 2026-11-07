# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a linear combination of 10 RBFs at fixed, evenly spaced centers.
    
    The model is:
    f(x) = sum_{j=0}^{9} w_j * exp(-(x - c_j)^2 / (2 * sigma^2))
    
    where c_j are fixed, evenly spaced centers in [-3, 3], and sigma is a single
    learnable width parameter shared by all kernels. The 10 parameters are:
    - params[0:9] = weights w_0 through w_8
    - params[9] = sigma (width parameter, clamped to [0.1, 3.0] for stability)
    
    This design uses:
    - Fixed, evenly spaced centers to ensure good coverage of the input domain.
    - A single shared width parameter to maintain a well-conditioned optimization
      problem while allowing adaptive smoothness.
    - Gaussian kernels for smooth, localized approximation capability.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Guard against non-finite inputs
    x = np.where(np.isfinite(x), x, 0.0)
    
    # Extract parameters
    w = params[:10]
    
    # Ensure weights are finite
    w = np.where(np.isfinite(w), w, 0.0)
    
    # Extract and clamp sigma (width parameter)
    sigma = w[9]
    sigma = np.clip(sigma, 0.1, 3.0)
    
    # Fixed, evenly spaced centers in [-3, 3]
    # 10 centers: -3, -2.333, -1.667, -1, -0.333, 0.333, 1, 1.667, 2.333, 3
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Compute RBF values: shape (N, 10)
    # diff[i, j] = x[i] - centers[j]
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Compute exp(-(x - c_j)^2 / (2 * sigma^2))
    exponent = -(diff ** 2) / (2.0 * sigma ** 2)
    
    # Clip exponent to avoid underflow/overflow issues
    exponent = np.clip(exponent, -50.0, 50.0)
    
    rbf_values = np.exp(exponent)
    
    # Compute prediction: y = rbf_values @ w[0:9]
    # Note: we use the first 9 weights for the 10 RBFs? 
    # Wait, we have 10 RBFs and 10 params. Let's use all 10 weights.
    # params[0:10] are the weights, but we used params[9] for sigma.
    # This is a conflict. We need 10 weights + 1 sigma = 11 params, but we only have 10.
    
    # Revised plan: Use 9 RBFs with 9 weights + 1 sigma = 10 params.
    # Or, use 10 RBFs but share one weight? No, let's use 9 RBFs.
    
    # Let's redo with 9 RBFs to fit the 10-param constraint (9 weights + 1 sigma).
    # Centers for 9 RBFs:
    centers = np.linspace(-3.0, 3.0, 9)
    
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    exponent = -(diff ** 2) / (2.0 * sigma ** 2)
    exponent = np.clip(exponent, -50.0, 50.0)
    rbf_values = np.exp(exponent)  # shape (N, 9)
    
    # Use the first 9 params as weights
    weights = w[:9]
    
    result = rbf_values @ weights
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
