"""Piecewise-linear phase-plane model with RBF basis for nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Piecewise-linear phase-plane model using RBF basis functions.
    
    Uses Gaussian RBFs centered in the (x, v) phase plane to create local
    linear patches, plus global sinusoidal drive terms. This captures
    non-additive x-v coupling that global polynomials miss.
    
    Inputs:
        t: time (range [30, 50])
        x: position (range [-0.2056, 0.1749])
        v: velocity (range [-0.3391, 0.3624])
        params: 10 fitted constants
    
    Returns:
        dv/dt (acceleration) for each row
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    p = np.asarray(params, dtype=np.float64).ravel()
    n = len(t)
    
    # Phase-plane centers for RBFs (covering the training range)
    # x range: [-0.2056, 0.1749], v range: [-0.3391, 0.3624]
    # Use 3 centers in each dimension for local structure
    x_centers = np.array([-0.15, 0.0, 0.12])
    v_centers = np.array([-0.25, 0.0, 0.25])
    
    # RBF bandwidth (inverse of width)
    sigma_x = 0.15
    sigma_v = 0.20
    
    # Build basis: 1 bias + 1 sin(t) + 1 cos(t) + 3 RBFs in (x,v) plane
    # That's 6 basis functions, leaving 4 params for local linear terms
    # Actually, let's use: bias, sin(t), cos(t), and 7 RBF-based local terms
    
    # RBF values: 3x3 grid = 9 RBFs, but we only have 10 params total
    # Use: bias(1) + sin(t)(1) + cos(t)(1) + 7 RBFs = 10
    
    # Select 7 RBF centers from the 3x3 grid (skip corners to save params)
    # Centers: (x_i, v_j) for i,j in range(3), skip two corners
    rbf_indices = [(0,1), (1,0), (1,1), (1,2), (2,1), (0,0), (2,2)]
    
    basis = np.empty((n, 10), dtype=np.float64)
    basis[:, 0] = 1.0  # bias
    basis[:, 1] = np.sin(t)  # drive
    basis[:, 2] = np.cos(t)  # drive quadrature
    
    for idx, (i, j) in enumerate(rbf_indices):
        xc = x_centers[i]
        vc = v_centers[j]
        dx = (x - xc) / sigma_x
        dv = (v - vc) / sigma_v
        rbf = np.exp(-0.5 * (dx**2 + dv**2))
        basis[:, 3 + idx] = rbf
    
    # Linear combination
    result = basis @ p
    
    # Safety: replace non-finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result

# EVOLVE-BLOCK-END
