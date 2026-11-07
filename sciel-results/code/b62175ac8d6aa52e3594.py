"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF (Radial Basis Function) interpolant for E. coli growth rate.
    Uses 10 fixed centers in the normalized 4D input space.
    The 10 parameters are the weights for each Gaussian basis function.
    
    Centers are placed at:
    - 4 corners of the hypercube
    - 4 mid-edges
    - 1 centroid
    - 1 additional center
    """
    # Normalize inputs to [0, 1] range based on training data ranges
    # b: [1, 4.998] -> [0, 1]
    # s: [1, 39.98] -> [0, 1]  
    # temp: [1, 12] -> [0, 1]
    # pH: [1, 4.999] -> [0, 1]
    
    b_norm = (b - 1.0) / (4.998 - 1.0)
    s_norm = (s - 1.0) / (39.98 - 1.0)
    temp_norm = (temp - 1.0) / (12.0 - 1.0)
    ph_norm = (pH - 1.0) / (4.999 - 1.0)
    
    # Stack normalized inputs into a matrix of shape (n_samples, 4)
    X = np.stack([b_norm, s_norm, temp_norm, ph_norm], axis=-1)
    
    # Define 10 fixed centers in the normalized space
    # Corners
    c0 = np.array([0.0, 0.0, 0.0, 0.0])
    c1 = np.array([1.0, 1.0, 1.0, 1.0])
    c2 = np.array([0.0, 1.0, 0.0, 1.0])
    c3 = np.array([1.0, 0.0, 1.0, 0.0])
    # Mid-edges
    c4 = np.array([0.5, 0.0, 0.5, 0.0])
    c5 = np.array([0.0, 0.5, 0.0, 0.5])
    c6 = np.array([1.0, 0.5, 1.0, 0.5])
    c7 = np.array([0.5, 1.0, 0.5, 1.0])
    # Centroid
    c8 = np.array([0.5, 0.5, 0.5, 0.5])
    # Additional center (slightly off-center for flexibility)
    c9 = np.array([0.3, 0.7, 0.6, 0.4])
    
    centers = np.array([c0, c1, c2, c3, c4, c5, c6, c7, c8, c9])
    
    # Compute squared distances from each sample to each center
    # X has shape (n_samples, 4), centers has shape (10, 4)
    # diff has shape (n_samples, 10, 4)
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    
    # Squared Euclidean distances, shape (n_samples, 10)
    sq_dist = np.sum(diff ** 2, axis=-1)
    
    # Gaussian RBF: exp(-beta * sq_dist)
    # Use a moderate beta for smooth interpolation
    beta = 4.0
    
    # Basis function values, shape (n_samples, 10)
    basis = np.exp(-beta * sq_dist)
    
    # Weighted sum: params has shape (10,)
    # prediction = basis @ params, shape (n_samples,)
    mu = basis @ params
    
    return mu
# EVOLVE-BLOCK-END
