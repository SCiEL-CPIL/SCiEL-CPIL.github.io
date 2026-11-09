"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression basis with 10 parameters.
    Combines polynomial, trigonometric, and mixed terms to approximate
    a wide range of smooth functions on [-1, 1]^2.
    
    The evaluator fits params via BFGS. We use a diverse set of 10 basis
    functions that cover linear, quadratic, and basic trigonometric behavior.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # 10 diverse basis functions:
    # 1. Constant
    # 2. x
    # 3. y
    # 4. x^2
    # 5. y^2
    # 6. x*y
    # 7. sin(x)
    # 8. sin(y)
    # 9. cos(x)
    # 10. cos(y)
    
    b1 = np.ones_like(x)
    b2 = x
    b3 = y
    b4 = x * x
    b5 = y * y
    b6 = x * y
    b7 = np.sin(x)
    b8 = np.sin(y)
    b9 = np.cos(x)
    b10 = np.cos(y)
    
    return (params[0] * b1 + 
            params[1] * b2 + 
            params[2] * b3 + 
            params[3] * b4 + 
            params[4] * b5 + 
            params[5] * b6 + 
            params[6] * b7 + 
            params[7] * b8 + 
            params[8] * b9 + 
            params[9] * b10)
# EVOLVE-BLOCK-END
