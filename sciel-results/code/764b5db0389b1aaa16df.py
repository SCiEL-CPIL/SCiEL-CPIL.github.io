# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT_LIMIT = 110  # Safety margin before 120s limit

# Fixed centers for Radial Basis Functions
# Covering the domain [-1, 1] x [-1, 1]
# We use 4 RBF centers to leave room for 6 Fourier terms
_CENTERS = np.array([
    [0.0, 0.0],
    [0.6, 0.6],
    [-0.6, 0.6],
    [0.6, -0.6]
])
_SIGMA = 0.6  # Width for RBFs

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 2 inputs x, y in [-1, 1]^2.
    
    Strategy: Hybrid model combining a Fourier basis and Radial Basis Functions (RBFs).
    The 10 parameters are split:
    - 6 parameters for Fourier basis: sin(x), cos(x), sin(y), cos(y), sin(x+y), cos(x-y)
    - 4 parameters for RBFs centered at 4 strategic points.
    
    The Fourier terms capture global periodic/oscillatory structure, while the RBFs
    provide local flexibility to capture non-periodic or complex features.
    This combination offers a richer function space than either alone, while remaining
    well-conditioned for BFGS optimization from an all-ones start.
    """
    # Deadline guard
    if time.time() - _start_time > _TIMEOUT_LIMIT:
        return np.full_like(y, np.mean(y))
        
    p = params
    
    # Fourier basis terms (6 params)
    f_sin_x = np.sin(x)
    f_cos_x = np.cos(x)
    f_sin_y = np.sin(y)
    f_cos_y = np.cos(y)
    f_sin_xy_plus = np.sin(x + y)
    f_cos_xy_minus = np.cos(x - y)
    
    # RBF terms (4 params)
    # x, y are arrays of shape (N,)
    # _CENTERS is shape (4, 2)
    
    dx = x[:, None] - _CENTERS[None, :, 0]
    dy = y[:, None] - _CENTERS[None, :, 1]
    
    dist_sq = dx**2 + dy**2
    
    # RBF values: exp(-dist_sq / (2 * sigma^2))
    # Shape: (N, 4)
    rbf_values = np.exp(-dist_sq / (2 * _SIGMA**2))
    
    # Combine:
    # Fourier part: (N,)
    fourier_part = (p[0] * f_sin_x +
                    p[1] * f_cos_x +
                    p[2] * f_sin_y +
                    p[3] * f_cos_y +
                    p[4] * f_sin_xy_plus +
                    p[5] * f_cos_xy_minus)
    
    # RBF part: (N, 4) @ (4,) -> (N,)
    rbf_part = rbf_values @ p[6:10]
    
    out = fourier_part + rbf_part
    
    return out
# EVOLVE-BLOCK-END
