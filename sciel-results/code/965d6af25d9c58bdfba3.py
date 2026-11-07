"""Symbolic regression with combined Chebyshev and trigonometric/exponential basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Compute y = sum_{i=0}^{9} params[i] * basis_i(x)
    
    Uses a hybrid basis:
    - Chebyshev polynomials T_0..T_5 on scaled domain [-1,1] for smooth polynomial approximation
    - sin(x), cos(x), exp(-x^2/9), tanh(x) for capturing periodic and localized features
    
    This combines the well-conditioned polynomial approximation of Chebyshev
    with the ability to capture non-polynomial functional forms.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure we have at least 10 params
    if p.shape[0] < 10:
        p = np.pad(p, (0, 10 - p.shape[0]), constant_values=0.0)
    
    # Scale x to [-1, 1] for Chebyshev polynomials
    z = x / 3.0
    
    # Compute Chebyshev polynomials T_0 ... T_5
    # T_0(z) = 1
    # T_1(z) = z
    # T_k(z) = 2*z*T_{k-1}(z) - T_{k-2}(z)
    
    T_prev2 = np.ones_like(z)  # T_0
    T_prev1 = z.copy()          # T_1
    
    result = p[0] * T_prev2
    result += p[1] * T_prev1
    
    for k in range(2, 6):
        T_curr = 2.0 * z * T_prev1 - T_prev2
        result += p[k] * T_curr
        T_prev2 = T_prev1
        T_prev1 = T_curr
        
    # Remaining basis functions: trigonometric and exponential
    result += p[6] * np.sin(x)       # sine
    result += p[7] * np.cos(x)       # cosine
    result += p[8] * np.exp(-x**2 / 9.0)  # Gaussian-like bump, centered at 0
    result += p[9] * np.tanh(x)      # hyperbolic tangent
    
    return result
# EVOLVE-BLOCK-END
