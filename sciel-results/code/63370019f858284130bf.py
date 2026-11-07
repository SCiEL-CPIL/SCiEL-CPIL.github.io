"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Parametric Fourier series with 10 basis functions.
    
    The 10 parameters are coefficients for:
    1. constant (1)
    2. x
    3. y
    4. sin(x)
    5. cos(x)
    6. sin(y)
    7. cos(y)
    8. sin(x+y)
    9. cos(x-y)
    10. sin(2*x)*cos(2*y)
    
    This is linear in the parameters, so BFGS converges to the global minimum.
    The Fourier basis can represent periodic and smooth functions on [-1,1]^2.
    """
    # Ensure x and y are 1D arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Build the 10 basis functions
    basis = np.column_stack([
        np.ones_like(x),          # p[0] * 1
        x,                         # p[1] * x
        y,                         # p[2] * y
        np.sin(x),                 # p[3] * sin(x)
        np.cos(x),                 # p[4] * cos(x)
        np.sin(y),                 # p[5] * sin(y)
        np.cos(y),                 # p[6] * cos(y)
        np.sin(x + y),             # p[7] * sin(x+y)
        np.cos(x - y),             # p[8] * cos(x-y)
        np.sin(2 * x) * np.cos(2 * y),  # p[9] * sin(2x)*cos(2y)
    ])
    
    # Linear combination
    pred = basis @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return pred
# EVOLVE-BLOCK-END
