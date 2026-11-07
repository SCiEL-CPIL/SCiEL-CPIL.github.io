"""
Direct approach: build a flexible model from the task structure.
The task asks for symbolic regression with 1 input on [-3, 3].
We design a basis that covers common smooth functions BFGS can fit with 10 params.
Uses Chebyshev polynomials for better conditioning.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Flexible model with 10 parameters using Chebyshev polynomials T0-T9.
    Chebyshev basis provides excellent conditioning for BFGS optimization,
    enabling reliable convergence to the global least-squares solution.
    """
    p = params
    x = np.asarray(x, dtype=np.float64)
    
    # Map x from [-3, 3] to [-1, 1] for Chebyshev stability
    x_scaled = x / 3.0
    
    # Compute Chebyshev polynomials T0 through T9 using recurrence:
    # T_0(x) = 1
    # T_1(x) = x
    # T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
    
    T = np.zeros((10, len(x_scaled)))
    T[0] = 1.0
    T[1] = x_scaled
    for n in range(2, 10):
        T[n] = 2 * x_scaled * T[n-1] - T[n-2]
    
    # Linear combination of Chebyshev basis functions
    result = T[0] * p[0] + T[1] * p[1] + T[2] * p[2] + T[3] * p[3] + T[4] * p[4]
    result += T[5] * p[5] + T[6] * p[6] + T[7] * p[7] + T[8] * p[8] + T[9] * p[9]
    
    return result
# EVOLVE-BLOCK-END
