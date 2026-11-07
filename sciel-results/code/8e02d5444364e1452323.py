"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 140.0  # safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Well-conditioned hybrid basis on [-3, 3]
    # Scale x to [-1, 1] for Chebyshev stability
    xs = x / 3.0
    
    # Chebyshev polynomials T_0..T_4 (well-conditioned)
    b0 = np.ones_like(x)
    b1 = xs
    b2 = 2.0 * xs * xs - 1.0
    b3 = 4.0 * xs * xs * xs - 3.0 * xs
    b4 = 8.0 * xs * xs * xs * xs - 8.0 * xs * xs + 1.0
    
    # Trigonometric basis (periodic components)
    b5 = np.sin(x)
    b6 = np.cos(x)
    b7 = np.sin(2.0 * x)
    b8 = np.cos(2.0 * x)
    
    # Modulated basis (captures x*sin(x)-type behavior)
    b9 = x * np.sin(x)
    
    # Linear combination
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
