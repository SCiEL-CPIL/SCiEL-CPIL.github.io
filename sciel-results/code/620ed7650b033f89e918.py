# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 2 inputs (x, y) drawn from [-1, 1]^2.
    
    Strategy: Gaussian RBF (Radial Basis Function) network with fixed centers.
    - 9 params for weights of 9 fixed Gaussian centers distributed across the domain.
    - 1 param for a global constant offset.
    
    The RBF basis provides local flexibility to model arbitrary smooth functions,
    complementing the global periodic trends that Fourier bases capture poorly.
    Fixed centers ensure stability and avoid parameter interactions that can
    hinder BFGS optimization.
    """
    p = params
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Fixed centers: 3x3 grid across [-1, 1]^2
    centers_x = np.array([-1.0, 0.0, 1.0])
    centers_y = np.array([-1.0, 0.0, 1.0])
    
    # Precompute distances from each point to each center
    # Shape: (n_points, 9)
    dists = np.zeros((x.shape[0], 9), dtype=np.float64)
    idx = 0
    for cx in centers_x:
        for cy in centers_y:
            dx = x - cx
            dy = y - cy
            dists[:, idx] = dx * dx + dy * dy
            idx += 1
    
    # Gaussian RBF with fixed bandwidth sigma = 0.5
    sigma = 0.5
    rbf_values = np.exp(-dists / (2.0 * sigma * sigma))
    
    # 9 RBF weights + 1 constant
    rbf_part = rbf_values @ p[:9]
    constant = p[9]
    
    return rbf_part + constant
# EVOLVE-BLOCK-END
