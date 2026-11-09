"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 150.0
_TIME_MARGIN = 5.0  # safety margin in seconds


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rich basis combining linear, centered quadratic, and non-linear terms.
    
    Strategy:
    - Use a diverse basis to help BFGS escape poor local minima.
    - Include sin and exp terms to capture non-linearities.
    - Keep conditioning reasonable by centering inputs.
    
    Basis allocation (10 params):
    p0: bias (1)
    p1: x
    p2: y
    p3: z
    p4: sin(x)
    p5: sin(y)
    p6: sin(z)
    p7: exp(-x)
    p8: exp(-y)
    p9: exp(-z)
    """
    # Check time budget
    if time.time() - _START_TIME > _TIME_LIMIT - _TIME_MARGIN:
        # Return a simple linear prediction as fallback
        return params[0] * x + params[1] * y + params[2] * z + params[3]
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    
    p = params
    
    # Build the prediction using diverse basis
    pred = (p[0] * np.ones_like(x) +
            p[1] * x +
            p[2] * y +
            p[3] * z +
            p[4] * np.sin(x) +
            p[5] * np.sin(y) +
            p[6] * np.sin(z) +
            p[7] * np.exp(-x) +
            p[8] * np.exp(-y) +
            p[9] * np.exp(-z))
    
    return pred
# EVOLVE-BLOCK-END
