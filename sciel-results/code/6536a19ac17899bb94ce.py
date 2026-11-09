"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Global rational function model using Chebyshev polynomials.
    f(x) = (sum_{i=0}^{4} a_i T_i(x)) / (sum_{j=0}^{4} b_j T_j(x))
    
    This can capture singularities and asymptotic behaviors that polynomial
    or trigonometric bases cannot. The denominator is regularized to avoid
    division by zero.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Map x from [-2, 2] to [-1, 1] for Chebyshev polynomials
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 through T_4
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    # T_4(t) = 8t^4 - 8t^2 + 1
    
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2 * t**2 - 1
    T3 = 4 * t**3 - 3 * t
    T4 = 8 * t**4 - 8 * t**2 + 1
    
    # Numerator: sum of a_i * T_i(x) for i=0..4
    # params[0:5] are the numerator coefficients
    numerator = (params[0] * T0 + 
                 params[1] * T1 + 
                 params[2] * T2 + 
                 params[3] * T3 + 
                 params[4] * T4)
    
    # Denominator: sum of b_j * T_j(x) for j=0..4
    # params[5:10] are the denominator coefficients
    # Add a small constant to ensure denominator is bounded away from zero
    denominator = (params[5] * T0 + 
                   params[6] * T1 + 
                   params[7] * T2 + 
                   params[8] * T3 + 
                   params[9] * T4)
    
    # Regularize denominator to avoid division by zero
    # Use a soft clamp: if |denom| < eps, replace with eps * sign(denom)
    eps = 1e-6
    denom_safe = np.where(np.abs(denominator) < eps, 
                          eps * np.sign(denominator + 1e-30), 
                          denominator)
    
    result = numerator / denom_safe
    
    return result
# EVOLVE-BLOCK-END
