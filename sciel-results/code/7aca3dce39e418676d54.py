"""Combined program: Chebyshev polynomial basis with non-linear exponential and sinusoidal components."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model combining:
    - Chebyshev polynomial basis (degrees 0-3) for well-conditioned global trend
    - Exponential decay term with variable rate for non-linear behavior
    - Sinusoidal term for oscillatory components
    
    y = sum_{k=0}^{3} p_k * T_k(2x/L - 1) 
      + p4 * exp(-p5 * x)
      + p6 * sin(p7 * x + p8)
      + p9 * T_4(2x/L - 1)
    
    where L = 6.0 (max domain from [0.1, 6])
    
    This combines the numerical conditioning benefits of Chebyshev polynomials
    with the non-linear expressive power of exponential and sinusoidal terms.
    """
    x = np.asarray(x, dtype=float)
    
    # Map x from [0.1, 6] to [-1, 1] for Chebyshev evaluation
    L = 6.0
    z = 2.0 * x / L - 1.0
    z = np.clip(z, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_0 through T_4 using the recurrence relation
    # T_0(z) = 1, T_1(z) = z, T_n(z) = 2*z*T_{n-1}(z) - T_{n-2}(z)
    T = np.zeros((5, x.shape[0]))
    T[0, :] = 1.0
    T[1, :] = z
    for n in range(2, 5):
        T[n, :] = 2.0 * z * T[n-1, :] - T[n-2, :]
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Chebyshev polynomial component (degrees 0-3)
    y_cheb = p0 * T[0] + p1 * T[1] + p2 * T[2] + p3 * T[3]
    
    # Additional Chebyshev term (degree 4)
    y_cheb4 = p9 * T[4]
    
    # Exponential decay term with variable rate
    # Clip exponent to avoid overflow/underflow during optimization
    exp_arg = -p5 * x
    exp_arg = np.clip(exp_arg, -50, 50)
    y_exp = p4 * np.exp(exp_arg)
    
    # Sinusoidal term with variable frequency and phase
    y_sin = p6 * np.sin(p7 * x + p8)
    
    return y_cheb + y_cheb4 + y_exp + y_sin
# EVOLVE-BLOCK-END
