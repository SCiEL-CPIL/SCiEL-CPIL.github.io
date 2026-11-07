import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Chebyshev polynomial expansion of degree 9.
    
    The model uses all 10 parameters as coefficients for Chebyshev polynomials
    T_0(x) through T_9(x), evaluated on the scaled domain [-1, 1] corresponding
    to x in [-2, 2].
    
    Chebyshev polynomials provide a stable, orthogonal-like basis that efficiently
    captures high-frequency oscillations and complex smooth shapes without the
    numerical instability associated with high-degree monomial polynomials.
    """
    # Scale x from [-2, 2] to [-1, 1]
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 through T_9 using recurrence
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n}(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    T0 = np.ones_like(t)
    T1 = t.copy()
    
    # Build the Chebyshev basis
    basis = [T0, T1]
    for n in range(2, 10):
        Tn = 2.0 * t * basis[-1] - basis[-2]
        basis.append(Tn)
    
    # Combine with parameters
    y = params[0] * basis[0]
    for i in range(1, 10):
        y = y + params[i] * basis[i]
    
    return y
