"""Symbolic regression with 1 input using a soft-thresholded Chebyshev basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: y = sum_{i=0}^{9} params[i] * basis_i(x)
    
    Basis functions (10 total):
    0: T_0(t) = 1
    1: T_1(t) = t
    2: T_2(t) = 2t^2 - 1
    3: T_3(t) = 4t^3 - 3t
    4: sin(x)
    5: cos(x)
    6: sin(2x)
    7: cos(2x)
    8: tanh(x)
    9: tanh(2x)
    
    where t = x / 3.
    
    The first four basis functions are standard Chebyshev polynomials of the
    first kind evaluated at t = x/3. The remaining six are trigonometric and
    saturating functions that capture periodic and step-like features.
    
    This design keeps the basis smooth and well-conditioned for BFGS while
    providing diverse functional forms to approximate a wide range of targets.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Scale x to [-1, 1] for Chebyshev polynomials
    t = x / 3.0
    
    # Chebyshev polynomials T_0 to T_3
    b0 = np.ones_like(x)
    b1 = t
    b2 = 2.0 * t**2 - 1.0
    b3 = 4.0 * t**3 - 3.0 * t
    
    # Trigonometric functions
    b4 = np.sin(x)
    b5 = np.cos(x)
    b6 = np.sin(2.0 * x)
    b7 = np.cos(2.0 * x)
    
    # Saturating functions (soft-step approximations)
    b8 = np.tanh(x)
    b9 = np.tanh(2.0 * x)
    
    # Stack into a matrix (N, 10)
    basis = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Linear combination
    y = basis @ params
    
    return y
# EVOLVE-BLOCK-END
