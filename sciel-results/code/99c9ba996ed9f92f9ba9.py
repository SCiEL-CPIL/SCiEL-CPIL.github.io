"""SEARCH: generate many candidate equations, fit each, keep the best."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 135.0  # safety margin before 150s kill

def _check_deadline():
    return time.time() - _start_time > _DEADLINE

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Structural Change: RBF Interpolant with 10 Fixed Centers.
    
    Uses 10 Gaussian RBF kernels with fixed centers at:
    - 4 corners of the normalized hypercube: [0,0,0,0], [1,1,1,1], [0,1,0,1], [1,0,1,0]
    - 4 mid-edges: [0.5,0,0,0], [0,0.5,0,0], [0,0,0.5,0], [0,0,0,0.5]
    - 1 centroid: [0.5,0.5,0.5,0.5]
    - 1 additional center: [0.5,0.5,0,0.5]
    
    Each RBF has a fixed width (sigma=0.5). The 10 parameters are the weights
    assigned to these basis functions. This provides a smooth, nonlinear model
    that can capture complex interactions without requiring polynomial terms.
    """
    # Normalize inputs to [0, 1] range for numerical stability
    b_n = (b - 1.0) / 4.0
    s_n = (s - 1.0) / 39.0
    t_n = (temp - 1.0) / 11.0
    p_n = (pH - 1.0) / 4.0
    
    # Stack normalized inputs into a 2D array: (n_samples, 4)
    inputs = np.column_stack([b_n, s_n, t_n, p_n])
    
    # Define 10 fixed centers
    centers = np.array([
        [0.0, 0.0, 0.0, 0.0],  # Corner 1
        [1.0, 1.0, 1.0, 1.0],  # Corner 2
        [0.0, 1.0, 0.0, 1.0],  # Corner 3
        [1.0, 0.0, 1.0, 0.0],  # Corner 4
        [0.5, 0.0, 0.0, 0.0],  # Mid-edge 1
        [0.0, 0.5, 0.0, 0.0],  # Mid-edge 2
        [0.0, 0.0, 0.5, 0.0],  # Mid-edge 3
        [0.0, 0.0, 0.0, 0.5],  # Mid-edge 4
        [0.5, 0.5, 0.5, 0.5],  # Centroid
        [0.5, 0.5, 0.0, 0.5]   # Additional center
    ])
    
    # Compute squared distances from each input to each center
    # inputs: (n_samples, 4), centers: (10, 4)
    # diff: (n_samples, 10, 4)
    diff = inputs[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=2)  # (n_samples, 10)
    
    # Compute RBF values: exp(-sq_dists / (2 * sigma^2))
    sigma = 0.5
    rbf_values = np.exp(-sq_dists / (2.0 * sigma ** 2))  # (n_samples, 10)
    
    # Linear combination: prediction = rbf_values @ params
    mu = rbf_values @ params
    
    return mu

# EVOLVE-BLOCK-END
