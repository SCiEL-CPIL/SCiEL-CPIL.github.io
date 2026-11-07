"""
Structural change: Global basis using Chebyshev polynomials mapped to the domain.
Uses 10 Chebyshev polynomials of the first kind (degree 0 to 9) as basis functions.
The model is linear in the 10 parameters (coefficients).
BFGS finds the global optimum for these coefficients.
"""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev polynomial model: 10 Chebyshev basis functions, 10 coefficients.
    x is mapped to [-1, 1] to use standard Chebyshev polynomials.
    
    params: array of length 10, each is the coefficient for one Chebyshev polynomial.
    """
    # Map x from [0.1, 6] to [-1, 1]
    # t = (2*x - (max+min)) / (max-min)
    x_min = 0.1
    x_max = 6.0
    t = (2.0 * x - (x_max + x_min)) / (x_max - x_min)
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    n_basis = 10
    basis = np.zeros((len(x), n_basis))
    
    if n_basis > 0:
        basis[:, 0] = 1.0
    if n_basis > 1:
        basis[:, 1] = t
        
    for i in range(2, n_basis):
        basis[:, i] = 2.0 * t * basis[:, i-1] - basis[:, i-2]
        
    # Output: weighted sum of basis functions
    prediction = basis @ params
    
    return prediction
