# EVOLVE-BLOCK-START
import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Local Linearization with Gaussian RBFs in the (x, v) phase plane.
    
    Structure:
    - 3 fixed Gaussian centers spread across the observed (x, v) range.
    - Each center provides a local linear model: w0 + w1*x + w2*v.
    - One global bias term for baseline offset.
    - This captures non-additive nonlinear coupling locally without
      relying on global polynomial or Fourier basis functions.
    
    Parameters (10 total):
    - params[0:3]:  Local linear weights for center 1 (bias, x-coeff, v-coeff)
    - params[3:6]:  Local linear weights for center 2 (bias, x-coeff, v-coeff)
    - params[6:9]:  Local linear weights for center 3 (bias, x-coeff, v-coeff)
    - params[9]:    Global bias
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    # Fixed Gaussian centers covering the observed phase plane range
    # x in [-0.2056, 0.1749], v in [-0.3391, 0.3624]
    centers = np.array([
        [-0.10, -0.15],   # lower-left region
        [ 0.00,  0.00],   # center
        [ 0.10,  0.15]    # upper-right region
    ])
    
    sigma = 0.18
    sigma2 = 2.0 * sigma * sigma
    
    # Compute 3 Gaussians: shape (3, N)
    dx = x[None, :] - centers[:, 0:1]  # (3, N)
    dv = v[None, :] - centers[:, 1:2]  # (3, N)
    g = np.exp(-(dx**2 + dv**2) / sigma2)  # (3, N)
    
    # Local linear predictions at each center
    # For each center k: g[k] * (params[3k] + params[3k+1]*x + params[3k+2]*v)
    local_pred = np.zeros_like(x, dtype=np.float64)
    for k in range(3):
        local_pred += g[k] * (params[3*k] + params[3*k+1] * x + params[3*k+2] * v)
    
    # Global bias
    pred = local_pred + params[9]
    
    return pred
# EVOLVE-BLOCK-END
