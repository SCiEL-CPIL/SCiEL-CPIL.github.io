import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Chebyshev polynomial expansion of degree 9.
    
    The model uses all 10 parameters as coefficients for Chebyshev polynomials
    T_0(x) through T_9(x). Chebyshev polynomials provide a stable, orthogonal-like
    basis on [-1, 1] that efficiently captures high-frequency oscillations and
    complex smooth shapes without the numerical instability of monomial polynomials.
    
    Since x is drawn from [-2, 2], we rescale to [-1, 1] for the Chebyshev basis.
    """
    # Rescale x from [-2, 2] to [-1, 1]
    t = 0.5 * x
    
    # Compute Chebyshev polynomials T_0 through T_9
    # T_0(x) = 1
    # T_1(x) = x
    # T_{n}(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
    
    T0 = np.ones_like(t)
    T1 = t.copy()
    
    y = params[0] * T0 + params[1] * T1
    
    if len(params) > 2:
        T2 = 2.0 * t * T1 - T0
        y += params[2] * T2
    
    if len(params) > 3:
        T3 = 2.0 * t * T2 - T1
        y += params[3] * T3
    
    if len(params) > 4:
        T4 = 2.0 * t * T3 - T2
        y += params[4] * T4
    
    if len(params) > 5:
        T5 = 2.0 * t * T4 - T3
        y += params[5] * T5
    
    if len(params) > 6:
        T6 = 2.0 * t * T5 - T4
        y += params[6] * T6
    
    if len(params) > 7:
        T7 = 2.0 * t * T6 - T5
        y += params[7] * T7
    
    if len(params) > 8:
        T8 = 2.0 * t * T7 - T6
        y += params[8] * T8
    
    if len(params) > 9:
        T9 = 2.0 * t * T8 - T7
        y += params[9] * T9
    
    return y
