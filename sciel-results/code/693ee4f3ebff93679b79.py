"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF (Radial Basis Function) model with 10 Gaussian basis functions.
    Centers are placed at corners and mid-edges of the normalized input space.
    The 10 parameters are the weights for each Gaussian kernel.
    
    Normalized inputs range approximately [-1, 1] for each dimension.
    Centers (10 total):
    0: centroid (0, 0, 0, 0)
    1-4: face centers along each axis
    5-8: edge midpoints (pairs of axes)
    9: corner (1, 1, 1, 1)
    """
    # Normalize inputs to [-1, 1] range
    b_norm = (b - 2.5) / 2.0
    s_norm = (s - 20.0) / 20.0
    temp_norm = (temp - 20.0) / 10.0
    pH_norm = (pH - 7.0) / 5.0
    
    # Stack normalized inputs
    X = np.stack([b_norm, s_norm, temp_norm, pH_norm], axis=-1)
    
    # Define 10 centers in normalized space
    centers = np.array([
        [0.0, 0.0, 0.0, 0.0],      # 0: centroid
        [0.0, 1.0, 0.0, 0.0],      # 1: s face center
        [0.0, -1.0, 0.0, 0.0],     # 2: s face center (opposite)
        [0.0, 0.0, 1.0, 0.0],      # 3: temp face center
        [0.0, 0.0, -1.0, 0.0],     # 4: temp face center (opposite)
        [0.0, 0.0, 0.0, 1.0],      # 5: pH face center
        [0.0, 0.0, 0.0, -1.0],     # 6: pH face center (opposite)
        [1.0, 0.0, 0.0, 0.0],      # 7: b face center
        [-1.0, 0.0, 0.0, 0.0],     # 8: b face center (opposite)
        [0.5, 0.5, 0.5, 0.5],      # 9: diagonal corner
    ])
    
    # Gaussian bandwidth (width parameter)
    sigma = 0.8
    
    # Compute RBF values: phi_i(x) = exp(-||x - c_i||^2 / (2 * sigma^2))
    # Shape: (n_samples, 10)
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (n, 10, 4)
    sq_dists = np.sum(diff ** 2, axis=2)  # (n, 10)
    phi = np.exp(-sq_dists / (2.0 * sigma ** 2))  # (n, 10)
    
    # Weighted sum of RBFs
    mu = phi @ params
    
    return mu
# EVOLVE-BLOCK-END
