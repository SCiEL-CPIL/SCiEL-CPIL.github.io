"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate using a rich basis of functions.
    
    The basis includes:
    - Polynomial terms: 1, x, x^2, x^3, x^4, x^5
    - Trigonometric terms: sin(x), cos(x)
    - Exponential terms: exp(x), exp(-x)
    - Rational terms: 1/(x^2+1)
    - Absolute value: |x| (approximated smoothly to keep differentiability for BFGS)
    
    We use 10 parameters as required. To ensure differentiability for BFGS,
    we replace |x| with x * tanh(k*x) where k is a fixed constant (e.g., 10).
    Actually, let's stick to standard smooth functions to avoid any non-differentiability issues.
    
    Basis:
    1. 1
    2. x
    3. x^2
    4. x^3
    4. x^4
    5. x^5
    6. sin(x)
    7. cos(x)
    8. exp(-x^2)  (Gaussian bump, smooth)
    9. 1/(1+x^2)  (Lorentzian, smooth)
    10. x * exp(-x^2)  (smooth odd function)
    
    This provides a diverse set of shapes that can approximate many common functions.
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis functions
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = x2
    b3 = x3
    b4 = x4
    b5 = x5
    b6 = np.sin(x)
    b7 = np.cos(x)
    b8 = np.exp(-x2)
    b9 = 1.0 / (1.0 + x2)
    
    # Combine
    return (p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + p4 * b4 + 
            p5 * b5 + p6 * b6 + p7 * b7 + p8 * b8 + p9 * b9)
# EVOLVE-BLOCK-END
