"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a linear combination of 9 Radial Basis Functions (RBFs)
    and a single learnable width parameter.
    
    params layout (length 10):
      p[0] to p[8]: Coefficients for the 9 RBFs
      p[9]:         Raw parameter for the RBF width (sigma)
    
    The RBFs are Gaussian kernels:
      phi_i(x) = exp(-(x - c_i)^2 / (2 * sigma^2))
    
    Centers c_i are evenly spaced across [-3, 3]:
      c_i = -3.0 + i * 0.75 for i in 0..8
    
    The width sigma is mapped from p[9] to [0.2, 2.5] to ensure numerical stability
    and smoothness for the BFGS optimizer.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.array([])
    
    # Ensure params is 1D array of length 10
    p = np.asarray(params, dtype=np.float64).flatten()
    if p.shape[0] != 10:
        # If wrong length, pad or truncate
        if p.shape[0] < 10:
            p = np.concatenate([p, np.ones(10 - p.shape[0])])
        else:
            p = p[:10]

    # Define centers for the 9 RBFs
    centers = np.array([-3.0, -2.25, -1.5, -0.75, 0.0, 0.75, 1.5, 2.25, 3.0])
    
    # Calculate width sigma from p[9]
    # Map tanh(p[9]) to [0.2, 2.5]
    # tanh range is [-1, 1].
    # 0.2 + (2.5 - 0.2) * (tanh(p[9]) + 1) / 2
    # = 0.2 + 1.15 * (tanh(p[9]) + 1)
    # = 1.35 + 1.15 * tanh(p[9])
    # Let's use a simpler mapping: 1.0 + 1.0 * tanh(p[9]) -> [0, 2]. 
    # But we need min 0.2 to avoid division by zero issues or overly spiky functions.
    # Let's use 1.25 + 1.05 * tanh(p[9]) -> [0.2, 2.3].
    sigma = 1.25 + 1.05 * np.tanh(p[9])
    
    # Pre-calculate 1/(2*sigma^2)
    inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma)
    
    # Initialize output
    y = np.zeros_like(x)
    
    # Compute linear combination of RBFs
    # p[0:9] are the coefficients
    for i in range(9):
        # Calculate (x - c_i)^2
        diff = x - centers[i]
        sq_diff = diff * diff
        # Calculate exp(-sq_diff * inv_two_sigma_sq)
        kernel = np.exp(-sq_diff * inv_two_sigma_sq)
        y += p[i] * kernel
        
    # Ensure finite output
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y
# EVOLVE-BLOCK-END
