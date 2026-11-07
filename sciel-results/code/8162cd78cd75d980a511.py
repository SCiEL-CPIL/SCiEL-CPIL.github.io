"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Product-based feature expansion: Chebyshev * nonlinear activation
    # This provides a globally stable basis with nonlinear modulation
    # to capture both periodic and non-periodic smooth functions.
    
    # Scale x to [-1, 1] for Chebyshev stability
    xs = x / 3.0
    
    # Chebyshev polynomials T_0..T_3 (well-conditioned on [-1,1])
    T0 = np.ones_like(x)
    T1 = xs
    T2 = 2.0 * xs * xs - 1.0
    T3 = 4.0 * xs * xs * xs - 3.0 * xs
    
    # Nonlinear activations
    # tanh for smooth sigmoid-like behavior
    tanh_x = np.tanh(x)
    tanh_2x = np.tanh(2.0 * x)
    
    # sin for periodic components
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    
    # Product features: Chebyshev * activation
    # b0: constant
    b0 = T0
    # b1: linear (T1 * 1)
    b1 = T1
    # b2: quadratic Chebyshev
    b2 = T2
    # b3: cubic Chebyshev
    b3 = T3
    # b4: T1 * tanh(x) - captures x*sigmoid behavior
    b4 = T1 * tanh_x
    # b5: T2 * tanh(x) - captures quadratic*sigmoid
    b5 = T2 * tanh_x
    # b6: T1 * sin(x) - captures x*periodic
    b6 = T1 * sin_x
    # b7: T2 * cos(x) - captures quadratic*periodic
    b7 = T2 * cos_x
    # b8: T1 * tanh(2x) - steeper sigmoid modulation
    b8 = T1 * tanh_2x
    # b9: T2 * sin(2x) - captures higher-frequency periodic
    b9 = T2 * np.sin(2.0 * x)
    
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
