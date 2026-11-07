"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis combining Gaussian Radial Basis Functions (RBFs) and low-degree polynomials.
    
    The Gaussian terms allow the model to capture local non-linear features and bumps,
    while the polynomial terms capture global trends. This is often more effective for
    unknown smooth functions than pure polynomial expansions when parameter count is limited,
    as it provides localized flexibility.
    
    Basis functions:
    1. exp(-(x^2 + y^2))
    2. exp(-((x-0.5)^2 + (y-0.5)^2))
    3. exp(-((x+0.5)^2 + (y+0.5)^2))
    4. exp(-((x-0.5)^2 + (y+0.5)^2))
    5. exp(-((x+0.5)^2 + (y-0.5)^2))
    6. 1
    7. x
    8. y
    9. x*y
    10. x^2 + y^2
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Calculate squared distances for Gaussian terms
    # Use a moderate width (sigma^2 = 1.0) to balance locality and smoothness
    # r^2 = (x-cx)^2 + (y-cy)^2
    
    # Center 1: (0, 0)
    r2_c0 = x*x + y*y
    b0 = np.exp(-r2_c0)
    
    # Center 2: (0.5, 0.5)
    r2_c1 = (x - 0.5)**2 + (y - 0.5)**2
    b1 = np.exp(-r2_c1)
    
    # Center 3: (-0.5, -0.5)
    r2_c2 = (x + 0.5)**2 + (y + 0.5)**2
    b2 = np.exp(-r2_c2)
    
    # Center 4: (0.5, -0.5)
    r2_c3 = (x - 0.5)**2 + (y + 0.5)**2
    b3 = np.exp(-r2_c3)
    
    # Center 5: (-0.5, 0.5)
    r2_c4 = (x + 0.5)**2 + (y - 0.5)**2
    b4 = np.exp(-r2_c4)
    
    # Polynomial terms
    b5 = np.ones_like(x)
    b6 = x
    b7 = y
    b8 = x * y
    b9 = x*x + y*y
    
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
