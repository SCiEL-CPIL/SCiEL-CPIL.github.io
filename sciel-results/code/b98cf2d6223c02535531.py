"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Polynomials + Trigonometric + Gaussian.
    This basis is designed to capture a wide range of functional forms
    including polynomials, periodic functions, and localized bumps.
    """
    # Ensure x is a numpy array for consistent operations
    x = np.asarray(x, dtype=np.float64)
    
    # Basis functions
    # 1-5: Polynomials (using powers for simplicity, stable enough for [-2,2])
    b0 = np.ones_like(x)
    b1 = x
    b2 = x**2
    b3 = x**3
    b4 = x**4
    
    # 6-9: Trigonometric functions (fundamental and 2nd harmonic)
    # Scaling x slightly? No, x is in [-2,2], so sin(x) covers ~1.5 radians.
    # If the function has higher frequency, we might miss it, but 
    # sin(2x) covers ~3 radians. 
    # Let's stick to sin(x), cos(x), sin(2x), cos(2x).
    b5 = np.sin(x)
    b6 = np.cos(x)
    b7 = np.sin(2.0 * x)
    b8 = np.cos(2.0 * x)
    
    # 10: Gaussian bump centered at 0
    # This is useful if the function has a peak or valley at 0.
    b9 = np.exp(-x**2)
    
    # Linear combination
    # params is an array of length 10
    return (params[0] * b0 +
            params[1] * b1 +
            params[2] * b2 +
            params[3] * b3 +
            params[4] * b4 +
            params[5] * b5 +
            params[6] * b6 +
            params[7] * b7 +
            params[8] * b8 +
            params[9] * b9)
# EVOLVE-BLOCK-END
