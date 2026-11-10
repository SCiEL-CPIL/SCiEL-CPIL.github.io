"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Global rational function model using Chebyshev polynomials:
    f(x) = N(x) / D(x)
    N(x) = sum_{i=0}^{4} a_i T_i(x)
    D(x) = 1 + sum_{j=1}^{5} b_j T_j(x)
    
    params[0:5] are numerator coefficients a_0..a_4
    params[5:10] are denominator coefficients b_1..b_5
    """
    x = np.asarray(x, dtype=np.float64)
    original_shape = x.shape
    x_flat = x.reshape(-1)
    
    # Compute Chebyshev polynomials T_0 through T_5
    T0 = np.ones_like(x_flat)
    T1 = x_flat.copy()
    T2 = 2.0 * x_flat * x_flat - 1.0
    T3 = 2.0 * x_flat * T2 - T1
    T4 = 2.0 * x_flat * T3 - T2
    T5 = 2.0 * x_flat * T4 - T3
    
    # Numerator coefficients
    a = params[0:5]
    numerator = a[0] * T0 + a[1] * T1 + a[2] * T2 + a[3] * T3 + a[4] * T4
    
    # Denominator coefficients
    # b_1..b_5 correspond to T_1..T_5
    b = params[5:10]
    # The constant term is fixed to 1 to remove scaling ambiguity and ensure stability near 0
    denominator = 1.0 + b[0] * T1 + b[1] * T2 + b[2] * T3 + b[3] * T4 + b[4] * T5
    
    # Ensure denominator doesn't get too close to zero
    eps = 1e-6
    abs_den = np.abs(denominator)
    sign = np.where(denominator >= 0, 1.0, -1.0)
    safe_denominator = np.where(abs_den < eps, eps * sign, denominator)
    
    result = numerator / safe_denominator
    
    # Clip to avoid extreme values
    result = np.clip(result, -1e4, 1e4)
    
    return result.reshape(original_shape)
# EVOLVE-BLOCK-END
