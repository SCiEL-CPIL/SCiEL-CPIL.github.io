"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression model using a Fourier-like basis.
    
    The basis is designed to be well-conditioned for BFGS optimization starting from all-ones.
    We use a combination of:
    - Constant term (params[0])
    - Linear term (params[1] * x)
    - Sine and cosine terms with increasing frequencies to capture periodic/non-linear behavior.
    
    x is in [0.1, 6]. We scale x to a more convenient range for the trigonometric functions.
    Let z = (x - 0.1) / (6 - 0.1) * 2 * pi -> maps [0.1, 6] to [0, 2*pi].
    This allows sine/cosine to capture full periods.
    
    Basis:
    b0 = 1
    b1 = x
    b2 = sin(z)
    b3 = cos(z)
    b4 = sin(2*z)
    b5 = cos(2*z)
    b6 = sin(3*z)
    b7 = cos(3*z)
    b8 = x**2
    b9 = x**3
    
    This provides a mix of polynomial and trigonometric flexibility.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    
    p = params
    
    # Scale x for trigonometric basis
    # Map [0.1, 6] to [0, 2*pi]
    z = (x - 0.1) / 5.9 * 2.0 * np.pi
    
    # Compute basis functions
    b0 = 1.0
    b1 = x
    b2 = np.sin(z)
    b3 = np.cos(z)
    b4 = np.sin(2.0 * z)
    b5 = np.cos(2.0 * z)
    b6 = np.sin(3.0 * z)
    b7 = np.cos(3.0 * z)
    b8 = x ** 2
    b9 = x ** 3
    
    # Linear combination
    y = (p[0] * b0 + 
         p[1] * b1 + 
         p[2] * b2 + 
         p[3] * b3 + 
         p[4] * b4 + 
         p[5] * b5 + 
         p[6] * b6 + 
         p[7] * b7 + 
         p[8] * b8 + 
         p[9] * b9)
    
    return y
# EVOLVE-BLOCK-END
