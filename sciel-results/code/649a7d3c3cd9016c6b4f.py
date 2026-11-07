"""Symbolic regression: nonlinear oscillator acceleration from position and velocity."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a Chebyshev polynomial basis
    in scaled position and velocity.
    
    The basis uses Chebyshev polynomials of the first kind evaluated at scaled
    inputs, providing superior approximation properties on bounded domains.
    
    params[0..9]: coefficients for the 10-term Chebyshev basis.
    """
    # Scale inputs to [-1, 1] range based on training ranges
    # x in [-0.5728, 0.5275] -> center ~ -0.0226, half-range ~ 0.5502
    # v in [-0.1754, 0.2821] -> center ~ 0.0534, half-range ~ 0.2288
    
    # Use symmetric scaling for stability
    x_center = -0.0226
    x_half = 0.5502
    v_center = 0.0534
    v_half = 0.2288
    
    # Scale to [-1, 1]
    X = (x - x_center) / x_half
    V = (v - v_center) / v_half
    
    # Clip to avoid numerical issues outside training range
    X = np.clip(X, -1.0, 1.0)
    V = np.clip(V, -1.0, 1.0)
    
    # Chebyshev polynomials of the first kind: T_n(cos(theta)) = cos(n*theta)
    # T_0 = 1
    # T_1 = x
    # T_2 = 2x^2 - 1
    # T_3 = 4x^3 - 3x
    # T_4 = 8x^4 - 8x^2 + 1
    
    # For 2D, we use a tensor-product-like basis but with only 10 terms.
    # We'll use a mixed basis:
    # Term 0: 1
    # Term 1: T_1(X) = X
    # Term 2: T_2(X) = 2X^2 - 1
    # Term 3: T_3(X) = 4X^3 - 3X
    # Term 4: T_1(V) = V
    # Term 5: T_2(V) = 2V^2 - 1
    # Term 6: T_3(V) = 4V^3 - 3V
    # Term 7: X * V
    # Term 8: X * V^2
    # Term 9: X^2 * V
    
    X2 = X * X
    X3 = X2 * X
    V2 = V * V
    V3 = V2 * V
    
    T1_X = X
    T2_X = 2 * X2 - 1
    T3_X = 4 * X3 - 3 * X
    
    T1_V = V
    T2_V = 2 * V2 - 1
    T3_V = 4 * V3 - 3 * V
    
    # Construct the 10-term basis
    basis_0 = np.ones_like(X)
    basis_1 = T1_X
    basis_2 = T2_X
    basis_3 = T3_X
    basis_4 = T1_V
    basis_5 = T2_V
    basis_6 = T3_V
    basis_7 = X * V
    basis_8 = X * V2
    basis_9 = X2 * V
    
    # Linear combination with params
    result = (params[0] * basis_0 +
              params[1] * basis_1 +
              params[2] * basis_2 +
              params[3] * basis_3 +
              params[4] * basis_4 +
              params[5] * basis_5 +
              params[6] * basis_6 +
              params[7] * basis_7 +
              params[8] * basis_8 +
              params[9] * basis_9)
    
    return result
# EVOLVE-BLOCK-END
