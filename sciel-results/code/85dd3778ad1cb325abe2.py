"""RBF-based symbolic regression with fixed centers in [-1,1]^2."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # RBF basis with 9 fixed Gaussian centers + 1 bias = 10 params
    # Centers: corners, edge midpoints, and center of [-1,1]^2
    centers = np.array([
        [-1.0, -1.0],  # p1
        [ 0.0, -1.0],  # p2
        [ 1.0, -1.0],  # p3
        [-1.0,  0.0],  # p4
        [ 0.0,  0.0],  # p5
        [ 1.0,  0.0],  # p6
        [-1.0,  1.0],  # p7
        [ 0.0,  1.0],  # p8
        [ 1.0,  1.0],  # p9
    ])
    
    # Gaussian bandwidth: sigma=0.6 gives good coverage of [-1,1]^2
    sigma = 0.6
    sigma2 = sigma * sigma
    
    # Compute squared distances from each point to each center
    # x, y are 1D arrays; centers shape (9, 2)
    dx = x[:, np.newaxis] - centers[:, 0][np.newaxis, :]  # (n, 9)
    dy = y[:, np.newaxis] - centers[:, 1][np.newaxis, :]  # (n, 9)
    dist2 = dx * dx + dy * dy  # (n, 9)
    
    # Compute RBF values: exp(-dist2 / (2*sigma2))
    rbf_vals = np.exp(-dist2 / (2.0 * sigma2))  # (n, 9)
    
    # params: [bias, w1, w2, ..., w9]
    bias = params[0]
    weights = params[1:]  # shape (9,)
    
    # Linear combination: bias + sum(w_i * rbf_i)
    pred = bias + rbf_vals @ weights
    
    return pred
# EVOLVE-BLOCK-END
