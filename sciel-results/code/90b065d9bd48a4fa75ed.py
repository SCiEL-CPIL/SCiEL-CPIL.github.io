"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a linear combination of
    Chebyshev polynomials, trigonometric functions, and a hyperbolic tangent.
    
    The input x is scaled to [-1, 1] to improve numerical stability.
    The parameters are fitted by the evaluator using BFGS.
    """
    # Scale x from [-2, 2] to [-1, 1]
    u = x / 2.0
    
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Chebyshev Polynomials T_0 to T_4
    # T_0(u) = 1
    t0 = 1.0
    # T_1(u) = u
    t1 = u
    # T_2(u) = 2u^2 - 1
    t2 = 2.0 * u**2 - 1.0
    # T_3(u) = 4u^3 - 3u
    t3 = 4.0 * u**3 - 3.0 * u
    # T_4(u) = 8u^4 - 8u^2 + 1
    t4 = 8.0 * u**4 - 8.0 * u**2 + 1.0
    
    # Trigonometric functions
    # Period 2 in u (Period 4 in x)
    s1 = np.sin(np.pi * u)
    c1 = np.cos(np.pi * u)
    # Period 1 in u (Period 2 in x)
    s2 = np.sin(2.0 * np.pi * u)
    c2 = np.cos(2.0 * np.pi * u)
    
    # Hyperbolic tangent
    th = np.tanh(u)
    
    # Combine
    y = (p0 * t0 + 
         p1 * t1 + 
         p2 * t2 + 
         p3 * t3 + 
         p4 * t4 + 
         p5 * s1 + 
         p6 * c1 + 
         p7 * s2 + 
         p8 * c2 + 
         p9 * th)
         
    return y
# EVOLVE-BLOCK-END
