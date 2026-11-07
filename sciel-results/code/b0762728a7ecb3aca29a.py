"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def _softplus(x, k, c):
    """Stable softplus: log(1 + exp(k * (x - c)))"""
    z = k * (x - c)
    # For large z, log(1 + exp(z)) approx z
    # For small z, log(1 + exp(z)) is stable
    # np.log1p(np.exp(z)) is stable for z < 0, but overflows for z > ~700
    # Given x in [-2, 2] and k up to 20, max z is 80. np.exp(80) is safe.
    return np.log1p(np.exp(z))


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure x is 1D for calculation, reshape at end if needed
    x_flat = np.asarray(x).flatten()
    
    # Basis 0: Constant
    b0 = np.ones_like(x_flat)
    
    # Basis 1: Linear
    b1 = x_flat
    
    # Basis 2-9: Softplus functions with fixed centers and slopes
    # Centers: -1.5, -0.5, 0.5, 1.5
    # Slopes: 5, 15
    centers = [-1.5, -0.5, 0.5, 1.5]
    slopes = [5.0, 15.0]
    
    basis_funcs = [b0, b1]
    
    for c in centers:
        for k in slopes:
            basis_funcs.append(_softplus(x_flat, k, c))
            
    # Stack into a matrix (N, 10)
    B = np.column_stack(basis_funcs)
    
    # Prediction is linear combination
    # params has length 10
    y_pred = B @ params
    
    # Reshape to match input shape
    return y_pred.reshape(np.asarray(x).shape)
# EVOLVE-BLOCK-END
