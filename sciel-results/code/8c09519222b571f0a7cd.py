"""Symbolic regression with 2 inputs using Chebyshev basis."""
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Chebyshev tensor-product basis.
    
    The 10 parameters are linear weights over 10 Chebyshev basis functions
    of total degree up to 3 in 2 variables. Chebyshev polynomials provide
    a well-conditioned basis for BFGS optimization, enabling reliable convergence.
    
    Basis functions (total degree <= 3):
    Degree 0: T0(x)T0(y)
    Degree 1: T1(x)T0(y), T0(x)T1(y)
    Degree 2: T2(x)T0(y), T1(x)T1(y), T0(x)T2(y)
    Degree 3: T3(x)T0(y), T2(x)T1(y), T1(x)T2(y), T0(x)T3(y)
    
    This gives exactly 1 + 2 + 3 + 4 = 10 basis functions.
    """
    # Chebyshev polynomials of the first kind
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    
    # Compute Chebyshev values for x and y
    T0_x = np.ones_like(x)
    T1_x = x.copy()
    T2_x = 2.0 * x**2 - 1.0
    T3_x = 4.0 * x**3 - 3.0 * x
    
    T0_y = np.ones_like(y)
    T1_y = y.copy()
    T2_y = 2.0 * y**2 - 1.0
    T3_y = 4.0 * y**3 - 3.0 * y
    
    # Build the 10 basis functions as columns of a design matrix
    # Order:
    # 0: T0(x)*T0(y)
    # 1: T1(x)*T0(y)
    # 2: T0(x)*T1(y)
    # 3: T2(x)*T0(y)
    # 4: T1(x)*T1(y)
    # 5: T0(x)*T2(y)
    # 6: T3(x)*T0(y)
    # 7: T2(x)*T1(y)
    # 8: T1(x)*T2(y)
    # 9: T0(x)*T3(y)
    
    basis = np.column_stack((
        T0_x * T0_y,
        T1_x * T0_y,
        T0_x * T1_y,
        T2_x * T0_y,
        T1_x * T1_y,
        T0_x * T2_y,
        T3_x * T0_y,
        T2_x * T1_y,
        T1_x * T2_y,
        T0_x * T3_y
    ))
    
    # Linear combination with the 10 parameters
    prediction = basis @ params
    
    return prediction
