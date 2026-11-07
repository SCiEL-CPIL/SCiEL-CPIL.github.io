"""Symbolic regression with 1 input(s) - Linear basis with Chebyshev and Trig terms."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a linear combination of stable basis functions.
    
    Basis functions:
    1. Chebyshev Polynomials T_0 to T_4 (stable polynomial approximation)
    2. Trigonometric functions sin(x), cos(x), sin(2x), cos(2x)
    3. Hyperbolic tangent tanh(x)
    
    The model is linear in the parameters, ensuring a convex loss surface for BFGS,
    which guarantees convergence to the global minimum for this specific basis.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # 1. Chebyshev Polynomials of the first kind
    # T_0(x) = 1
    # T_1(x) = x
    # T_2(x) = 2x^2 - 1
    # T_3(x) = 4x^3 - 3x
    # T_4(x) = 8x^4 - 8x^2 + 1
    
    T0 = np.ones_like(x)
    T1 = x
    T2 = 2 * x**2 - 1
    T3 = 4 * x**3 - 3 * x
    T4 = 8 * x**4 - 8 * x**2 + 1
    
    poly_part = p0 * T0 + p1 * T1 + p2 * T2 + p3 * T3 + p4 * T4
    
    # 2. Trigonometric Basis
    # Fixed frequencies to maintain smooth landscape for optimizer
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2 * x)
    cos_2x = np.cos(2 * x)
    
    trig_part = p5 * sin_x + p6 * cos_x + p7 * sin_2x + p8 * cos_2x
    
    # 3. Saturation Basis
    tanh_x = np.tanh(x)
    
    tanh_part = p9 * tanh_x
    
    return poly_part + trig_part + tanh_part
# EVOLVE-BLOCK-END
