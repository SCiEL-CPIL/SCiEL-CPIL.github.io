"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 115.0  # Safety margin for 120s limit

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(x)
    
    # Structural Change:
    # 1. Use Chebyshev polynomials T0-T4 on scaled x for well-conditioned polynomial basis
    # 2. Add trigonometric terms on raw x for periodic components
    # 3. Add two Gaussian RBFs at different centers for localized features
    # This gives a diverse, well-conditioned basis that BFGS can navigate effectively.
    
    z = x / 3.0  # Scale to [-1, 1] for Chebyshev
    
    # Chebyshev polynomials T0 through T4
    b0 = np.ones_like(x)
    b1 = z
    b2 = 2.0 * z ** 2 - 1.0
    b3 = 4.0 * z ** 3 - 3.0 * z
    b4 = 8.0 * z ** 4 - 8.0 * z ** 2 + 1.0
    
    # Trigonometric basis on raw x
    b5 = np.sin(x)
    b6 = np.cos(x)
    b7 = np.sin(2.0 * x)
    b8 = np.cos(2.0 * x)
    
    # Two Gaussian RBFs at different centers with fixed widths
    b9 = np.exp(-((x - 1.0) ** 2) / 1.0)
    
    y = (params[0] * b0 + 
         params[1] * b1 + 
         params[2] * b2 + 
         params[3] * b3 + 
         params[4] * b4 + 
         params[5] * b5 + 
         params[6] * b6 + 
         params[7] * b7 + 
         params[8] * b8 + 
         params[9] * b9)
         
    return y
# EVOLVE-BLOCK-END
