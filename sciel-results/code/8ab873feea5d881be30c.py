"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 1 input.
    Uses a multi-frequency Fourier basis plus a linear term.
    
    Basis functions:
    1. Constant
    2. x (linear trend)
    3. sin(x)
    4. cos(x)
    5. sin(2x)
    6. cos(2x)
    7. sin(3x)
    8. cos(3x)
    9. sin(4x)
    10. cos(4x)
    
    The params array has length 10, corresponding to coefficients for each basis function.
    """
    x = np.asarray(x, dtype=float)
    
    # Ensure x is 1D for consistency
    if x.ndim > 1:
        x = x.ravel()
    
    # Compute basis functions
    # Using a Fourier series up to frequency 4, plus a linear term
    basis = np.column_stack([
        np.ones_like(x),
        x,
        np.sin(x),
        np.cos(x),
        np.sin(2 * x),
        np.cos(2 * x),
        np.sin(3 * x),
        np.cos(3 * x),
        np.sin(4 * x),
        np.cos(4 * x),
    ])
    
    # Compute prediction: linear combination of basis functions
    # params has length 10
    params = np.asarray(params, dtype=float)
    if params.shape[0] != 10:
        # Pad or truncate if needed (shouldn't happen with correct usage)
        new_params = np.ones(10)
        n = min(params.shape[0], 10)
        new_params[:n] = params[:n]
        params = new_params
    
    prediction = basis @ params
    
    # Ensure finite values
    prediction = np.where(np.isfinite(prediction), prediction, 0.0)
    
    return prediction
# EVOLVE-BLOCK-END
