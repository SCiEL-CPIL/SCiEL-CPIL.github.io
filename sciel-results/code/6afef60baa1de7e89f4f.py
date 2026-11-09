"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 1 input.
    Uses a flexible basis of functions to approximate the target.
    
    Basis functions:
    1. Constant
    2. x
    3. x^2
    4. x^3
    5. sin(x)
    6. cos(x)
    7. exp(x)
    8. tanh(x)
    9. abs(x)
    10. x * sin(x)
    
    The params array has length 10, corresponding to coefficients for each basis function.
    """
    x = np.asarray(x, dtype=float)
    
    # Ensure x is 1D for consistency
    if x.ndim > 1:
        x = x.ravel()
    
    # Compute basis functions
    basis = np.column_stack([
        np.ones_like(x),
        x,
        x ** 2,
        x ** 3,
        np.sin(x),
        np.cos(x),
        np.exp(x),
        np.tanh(x),
        np.abs(x),
        x * np.sin(x),
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
