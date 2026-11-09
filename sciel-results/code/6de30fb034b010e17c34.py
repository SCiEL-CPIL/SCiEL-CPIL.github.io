"""Symbolic regression with Chebyshev polynomial basis for 2 inputs."""
import numpy as np
import time
import warnings


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using Chebyshev polynomials in x and y.
    
    The basis is:
      T_i(x) * T_j(y) for i+j <= 2 (degree up to 2 in total)
      This gives: T0(x)T0(y), T1(x)T0(y), T0(x)T1(y), T2(x)T0(y), 
                  T1(x)T1(y), T0(x)T2(y)
      That's 6 terms. We use 10 params, so we add higher degree terms:
      T2(x)T1(y), T1(x)T2(y), T3(x)T0(y), T0(x)T3(y)
      Total: 10 terms.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    
    # Compute Chebyshev polynomials up to degree 3
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2*t^2 - 1
    # T_3(t) = 4*t^3 - 3*t
    
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2.0 * x**2 - 1.0
    T3_x = 4.0 * x**3 - 3.0 * x
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2.0 * y**2 - 1.0
    T3_y = 4.0 * y**3 - 3.0 * y
    
    # Build the 10 basis functions
    # Order:
    # 0: T0(x)*T0(y)  = 1
    # 1: T1(x)*T0(y)  = x
    # 2: T0(x)*T1(y)  = y
    # 3: T2(x)*T0(y)  = 2x^2-1
    # 4: T1(x)*T1(y)  = x*y
    # 5: T0(x)*T2(y)  = 2y^2-1
    # 6: T2(x)*T1(y)  = (2x^2-1)*y
    # 7: T1(x)*T2(y)  = x*(2y^2-1)
    # 8: T3(x)*T0(y)  = 4x^3-3x
    # 9: T0(x)*T3(y)  = 4y^3-3y
    
    p = params
    
    result = (
        p[0] * T0_x * T0_y +
        p[1] * T1_x * T0_y +
        p[2] * T0_x * T1_y +
        p[3] * T2_x * T0_y +
        p[4] * T1_x * T1_y +
        p[5] * T0_x * T2_y +
        p[6] * T2_x * T1_y +
        p[7] * T1_x * T2_y +
        p[8] * T3_x * T0_y +
        p[9] * T0_x * T3_y
    )
    
    return result
