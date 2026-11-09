"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Global rational function model using Chebyshev polynomials:
    f(x) = (sum_{i=0}^{4} a_i T_i(x)) / (sum_{j=0}^{4} b_j T_j(x))
    
    Both numerator and denominator are degree-4 Chebyshev polynomials.
    This rational form can capture singularities and asymptotic behaviors.
    
    params[0:5] are numerator coefficients a_0..a_4
    params[5:10] are denominator coefficients b_0..b_4
    """
    x = np.asarray(x, dtype=np.float64)
    original_shape = x.shape
    x_flat = x.reshape(-1)
    
    # Compute Chebyshev polynomials T_0 through T_4
    T0 = np.ones_like(x_flat)
    T1 = x_flat.copy()
    T2 = 2.0 * x_flat * x_flat - 1.0
    T3 = 2.0 * x_flat * T2 - T1
    T4 = 2.0 * x_flat * T3 - T2
    
    # Numerator: a_0*T0 + a_1*T1 + a_2*T2 + a_3*T3 + a_4*T4
    a = params[0:5]
    numerator = a[0] * T0 + a[1] * T1 + a[2] * T2 + a[3] * T3 + a[4] * T4
    
    # Denominator: b_0*T0 + b_1*T1 + b_2*T2 + b_3*T3 + b_4*T4
    b = params[5:10]
    denominator = b[0] * T0 + b[1] * T1 + b[2] * T2 + b[3] * T3 + b[4] * T4
    
    # Ensure denominator doesn't get too close to zero for numerical stability
    eps = 1e-10
    abs_den = np.abs(denominator)
    sign = np.where(denominator >= 0, 1.0, -1.0)
    safe_denominator = np.where(abs_den < eps, eps * sign, denominator)
    
    result = numerator / safe_denominator
    
    # Clip to avoid extreme values that could cause numerical issues
    result = np.clip(result, -1e6, 1e6)
    
    return result.reshape(original_shape)
# EVOLVE-BLOCK-END
