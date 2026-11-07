"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev polynomial basis (total degree <= 3) with 10 parameters.
    Orthogonal basis improves conditioning for BFGS convergence.
    
    Basis functions (Chebyshev T_n):
    T0=1, T1=x, T2=2x^2-1, T3=4x^3-3x
    1. T0(x)*T0(y) = 1
    2. T1(x)*T0(y) = x
    3. T0(x)*T1(y) = y
    4. T2(x)*T0(y) = 2x^2-1
    5. T0(x)*T2(y) = 2y^2-1
    6. T1(x)*T1(y) = x*y
    7. T3(x)*T0(y) = 4x^3-3x
    8. T0(x)*T3(y) = 4y^3-3y
    9. T1(x)*T2(y) = x*(2y^2-1)
    10. T2(x)*T1(y) = (2x^2-1)*y
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Chebyshev polynomials
    T0x = np.ones_like(x)
    T1x = x
    T2x = 2.0 * x**2 - 1.0
    T3x = 4.0 * x**3 - 3.0 * x
    
    T0y = np.ones_like(y)
    T1y = y
    T2y = 2.0 * y**2 - 1.0
    T3y = 4.0 * y**3 - 3.0 * y
    
    # Basis vectors
    b0 = T0x * T0y
    b1 = T1x * T0y
    b2 = T0x * T1y
    b3 = T2x * T0y
    b4 = T0x * T2y
    b5 = T1x * T1y
    b6 = T3x * T0y
    b7 = T0x * T3y
    b8 = T1x * T2y
    b9 = T2x * T1y
    
    return (p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + p4 * b4 + 
            p5 * b5 + p6 * b6 + p7 * b7 + p8 * b8 + p9 * b9)
# EVOLVE-BLOCK-END
