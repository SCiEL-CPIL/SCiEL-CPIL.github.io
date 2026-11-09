# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF network with 4 centers, 4 weights, 1 offset, 1 width parameter.
    Params layout:
    0-3: Center shifts (added to base centers -3, -1, 1, 3)
    4-7: Weights
    8: Offset
    9: Inverse variance (width control)
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    c_shifts = params[0:4]
    weights = params[4:8]
    offset = params[8]
    inv_var = params[9]
    
    # Ensure inverse variance is positive to avoid division by zero or negative width
    # exp is a safe way to keep it positive. If inv_var is large, sigma is small.
    # If inv_var is small, sigma is large.
    # Starting inv_var = 1 -> exp(1) = 2.7 -> sigma = 1/sqrt(2.7) ~ 0.6.
    # Let's use a softer mapping or just clip.
    # Actually, if inv_var becomes negative, exp(inv_var) is still positive.
    k = np.exp(inv_var)
    
    # Base centers spread across [-3, 3]
    base_centers = np.array([-3.0, -1.0, 1.0, 3.0])
    centers = base_centers + c_shifts
    
    # Compute RBF outputs
    # x is (n,), centers is (4,)
    # diff is (n, 4)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian kernel: exp(-k * diff^2 / 2)
    # k is scalar
    rbf_vals = np.exp(-0.5 * k * diff**2)
    
    # Weighted sum
    pred = rbf_vals @ weights + offset
    
    return pred
# EVOLVE-BLOCK-END
