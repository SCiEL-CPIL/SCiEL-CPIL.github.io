"""Non-linear parametric model: Sum of Exponentials (RBF-like) with learned exponents."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a sum of 3 Gaussian-like exponential terms.
    
    Model:
    f(x, y, z) = c0 + sum_{i=1}^{3} c_i * exp(-w_i * || [x, y, z] - center_i ||^2)
    
    Parameters (length 10):
    - c0: params[0] (Global offset)
    - For each term i (1, 2, 3):
        - c_i: params[1 + 3*(i-1)] (Amplitude)
        - w_i: params[2 + 3*(i-1)] (Width/Sharpness, softplus transformed to ensure positive)
        - center_i: params[3 + 3*(i-1)] (Log-scale center, tanh mapped to [0.5, 2.0])
    
    This approach is materially different from a linear basis expansion because:
    1. The parameters define the *structure* (widths, centers) of the basis functions, not just their weights.
    2. It allows for localized features (peaks/valleys) that a global polynomial/exponential mix cannot easily capture with only 10 weights.
    3. BFGS can optimize the non-linear parameters directly.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    # Ensure params is a flat array of length 10
    params = np.asarray(params, dtype=np.float64).ravel()
    if params.size != 10:
        # Fallback if shape is wrong, though contract says length 10
        params = np.ones(10)

    # Extract parameters
    c0 = params[0]
    
    # We have 9 remaining params for 3 terms.
    # Term 1: params[1], params[2], params[3]
    # Term 2: params[4], params[5], params[6]
    # Term 3: params[7], params[8], params[9]
    
    prediction = np.full_like(x, c0, dtype=np.float64)
    
    for i in range(3):
        idx_c = 1 + 3 * i
        idx_w = 2 + 3 * i
        idx_k = 3 + 3 * i
        
        c_i = params[idx_c]
        w_raw = params[idx_w]
        k_raw = params[idx_k]
        
        # Transform width to be positive and stable
        # Softplus: log(1 + exp(w)) is smooth and positive.
        # To avoid overflow in exp for large w, we use a stable softplus approximation
        # or simply clamp. Since BFGS starts at 1, w_raw is likely moderate.
        # Stable softplus: max(w, 0) + log(1 + exp(-abs(w)))
        w_abs = np.abs(w_raw)
        w_pos = np.maximum(w_raw, 0) + np.log1p(np.exp(-w_abs))
        # Clamp w_pos to prevent extremely narrow spikes that cause numerical issues
        # or very wide ones that vanish.
        w_pos = np.clip(w_pos, 0.1, 10.0)
        
        # Transform center to [0.5, 2.0]
        # tanh(k) is in [-1, 1]. Map to [0.5, 2.0]: 1.25 + 0.75 * tanh(k)
        center = 1.25 + 0.75 * np.tanh(k_raw)
        
        # Calculate squared distance
        dx = x - center
        dy = y - center
        dz = z - center
        dist_sq = dx*dx + dy*dy + dz*dz
        
        # Calculate exponential term
        # exp(-w * dist_sq)
        # dist_sq is in [0, 3*(1.5)^2] = [0, 6.75] approx.
        # w is in [0.1, 10].
        # Exponent is in [-67.5, 0]. Safe.
        term = c_i * np.exp(-w_pos * dist_sq)
        
        prediction += term
    
    # Ensure finite output
    if not np.all(np.isfinite(prediction)):
        return np.full_like(x, np.nan)
        
    return prediction
# EVOLVE-BLOCK-END
