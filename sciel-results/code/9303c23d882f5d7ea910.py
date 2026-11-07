"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Universal approximator using 10 Gaussian Radial Basis Functions (RBFs).
    
    Strategy:
    Instead of a global polynomial basis, we use 10 local Gaussian bumps.
    The centers are fixed at a grid of points covering the domain [-1, 1]^2.
    The parameters control the amplitude (weight) of each bump.
    This allows fitting non-polynomial smooth functions (e.g., sin, exp, localized features)
    which a degree-3 polynomial cannot represent accurately.
    
    Basis functions:
    phi_i(x, y) = exp(-gamma * ((x - c_xi)^2 + (y - c_yi)^2))
    
    Centers (c_xi, c_yi) are chosen from a 3x3 grid plus one center, 
    or simply a 2x5 grid, or similar. 
    To maximize coverage with 10 points, let's use a specific set of centers:
    1. (-1, -1)
    2. (-1, 0)
    3. (-1, 1)
    4. (0, -1)
    5. (0, 0)
    6. (0, 1)
    7. (1, -1)
    8. (1, 0)
    9. (1, 1)
    10. (0, 0) -- wait, duplicate? 
    Let's use a 2x5 grid:
    x_centers: [-1, -0.5, 0, 0.5, 1]
    y_centers: [-1, 1]
    This gives 10 distinct centers.
    
    The width (sigma) is fixed to 1.0 for stability, allowing BFGS to focus on amplitudes.
    A constant term is implicitly handled by the overlap of Gaussians, but 
    to ensure we can fit a constant function easily, we could reserve one param for a bias.
    However, the sum of Gaussians can approximate a constant if weights are tuned.
    Given BFGS starts at 1, a constant function y=1 would be approximated by sum of weights * gaussians.
    To be safe, let's make the last basis function a constant 1.0.
    
    Basis:
    9 Gaussians at grid points + 1 Constant term.
    """
    p = params
    
    # Fixed centers for 9 Gaussians
    # Grid: x in [-1, 0, 1], y in [-1, 0, 1]
    cx = [-1.0, 0.0, 1.0]
    cy = [-1.0, 0.0, 1.0]
    
    # Precompute squared distances for the 9 centers
    # To be efficient, we compute this inside the function as x,y change per call
    # But since we only have 10 params, the computation is light.
    
    # Unpack weights for Gaussians
    w1, w2, w3, w4, w5, w6, w7, w8, w9 = p[0:9]
    # Last param is constant bias
    bias = p[9]
    
    # Calculate squared distances
    # Center 1: (-1, -1)
    d1 = (x - (-1.0))**2 + (y - (-1.0))**2
    # Center 2: (-1, 0)
    d2 = (x - (-1.0))**2 + (y - 0.0)**2
    # Center 3: (-1, 1)
    d3 = (x - (-1.0))**2 + (y - 1.0)**2
    # Center 4: (0, -1)
    d4 = (x - 0.0)**2 + (y - (-1.0))**2
    # Center 5: (0, 0)
    d5 = (x - 0.0)**2 + (y - 0.0)**2
    # Center 6: (0, 1)
    d6 = (x - 0.0)**2 + (y - 1.0)**2
    # Center 7: (1, -1)
    d7 = (x - 1.0)**2 + (y - (-1.0))**2
    # Center 8: (1, 0)
    d8 = (x - 1.0)**2 + (y - 0.0)**2
    # Center 9: (1, 1)
    d9 = (x - 1.0)**2 + (y - 1.0)**2
    
    # Gaussian kernel: exp(-d^2 / (2 * sigma^2))
    # Let sigma = 0.8, so 2*sigma^2 = 1.28. 
    # A width of 0.8 allows some overlap but keeps locality.
    # Using a slightly wider kernel (sigma=1.0 -> 2*sigma^2=2.0) might be safer for BFGS convergence from 1s.
    # Let's use sigma=1.0.
    inv_2_sigma2 = 0.5
    
    g1 = np.exp(-d1 * inv_2_sigma2)
    g2 = np.exp(-d2 * inv_2_sigma2)
    g3 = np.exp(-d3 * inv_2_sigma2)
    g4 = np.exp(-d4 * inv_2_sigma2)
    g5 = np.exp(-d5 * inv_2_sigma2)
    g6 = np.exp(-d6 * inv_2_sigma2)
    g7 = np.exp(-d7 * inv_2_sigma2)
    g8 = np.exp(-d8 * inv_2_sigma2)
    g9 = np.exp(-d9 * inv_2_sigma2)
    
    # Combine
    pred = (
        w1 * g1 + 
        w2 * g2 + 
        w3 * g3 + 
        w4 * g4 + 
        w5 * g5 + 
        w6 * g6 + 
        w7 * g7 + 
        w8 * g8 + 
        w9 * g9 + 
        bias
    )
    
    return pred
# EVOLVE-BLOCK-END
