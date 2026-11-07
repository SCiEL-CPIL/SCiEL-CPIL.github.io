# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid polynomial-Fourier-nonlinear basis model with 10 parameters.
    
    Basis functions (10 total, linear in params):
    0: 1 (constant)
    1: x
    2: x^2
    3: x^3
    4: x^4
    5: sin(x)
    6: cos(x)
    7: sin(2x)
    8: tanh(x)
    9: exp(-x^2/2)
    
    This basis covers:
    - Polynomials up to degree 4 for smooth trend fitting.
    - Fourier terms (1st and 2nd harmonics) for periodic components.
    - Tanh term for non-linear saturation or step-like behavior.
    - Gaussian term for localized bumps or decay.
    """
    p = params
    
    # Polynomial terms
    t0 = np.ones_like(x)
    t1 = x
    t2 = x * x
    t3 = t2 * x
    t4 = t2 * t2
    
    # Fourier terms
    t5 = np.sin(x)
    t6 = np.cos(x)
    t7 = np.sin(2.0 * x)
    
    # Non-linear saturation term
    t8 = np.tanh(x)
    
    # Gaussian decay term
    t9 = np.exp(-0.5 * t2)
    
    y = (p[0] * t0 +
         p[1] * t1 +
         p[2] * t2 +
         p[3] * t3 +
         p[4] * t4 +
         p[5] * t5 +
         p[6] * t6 +
         p[7] * t7 +
         p[8] * t8 +
         p[9] * t9)
    
    return y
# EVOLVE-BLOCK-END
