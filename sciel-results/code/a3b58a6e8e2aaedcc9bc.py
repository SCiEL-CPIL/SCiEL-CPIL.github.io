"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form bivariate function using a Fourier (trigonometric) basis.
    
    This basis is orthogonal on the domain [-1, 1] (up to scaling) and provides
    a different modelling family compared to polynomials. It is particularly
    effective for smooth functions and can capture oscillatory behavior.
    
    Basis functions:
    1. 1
    2. cos(x)
    3. sin(x)
    4. cos(y)
    5. sin(y)
    6. cos(2x)
    7. sin(2x)
    8. cos(2y)
    9. sin(2y)
    10. cos(x)cos(y)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Precompute trigonometric terms
    cos_x = np.cos(x)
    sin_x = np.sin(x)
    cos_y = np.cos(y)
    sin_y = np.sin(y)
    
    cos_2x = np.cos(2.0 * x)
    sin_2x = np.sin(2.0 * x)
    cos_2y = np.cos(2.0 * y)
    sin_2y = np.sin(2.0 * y)
    
    # 10 basis functions
    b0 = np.ones_like(x)       # 1
    b1 = cos_x                 # cos(x)
    b2 = sin_x                 # sin(x)
    b3 = cos_y                 # cos(y)
    b4 = sin_y                 # sin(y)
    b5 = cos_2x                # cos(2x)
    b6 = sin_2x                # sin(2x)
    b7 = cos_2y                # cos(2y)
    b8 = sin_2y                # sin(2y)
    b9 = cos_x * cos_y         # cos(x)cos(y)
    
    # Linear combination
    pred = (params[0] * b0 +
            params[1] * b1 +
            params[2] * b2 +
            params[3] * b3 +
            params[4] * b4 +
            params[5] * b5 +
            params[6] * b6 +
            params[7] * b7 +
            params[8] * b8 +
            params[9] * b9)
    
    # Ensure output is finite
    if not np.all(np.isfinite(pred)):
        return np.mean(y) * np.ones_like(x)
        
    return pred
# EVOLVE-BLOCK-END
