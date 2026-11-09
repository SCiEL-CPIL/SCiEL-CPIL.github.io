"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Global rational function model using Chebyshev polynomials:
    f(x) = (a_0 T_0 + a_1 T_1 + ... + a_4 T_4) / (1 + c_1 T_1 + c_2 T_2 + c_3 T_3 + c_4 T_4)
    
    The denominator is parameterized as 1 + sum(c_i * T_i) to ensure it's close to 1,
    improving numerical stability. The numerator uses unconstrained coefficients.
    
    params[0:5] are numerator coefficients a_0..a_4
    params[5:10] are denominator perturbation coefficients c_1..c_4 (and one unused or scaled)
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
    
    # Numerator coefficients
    a = params[0:5]
    numerator = a[0] * T0 + a[1] * T1 + a[2] * T2 + a[3] * T3 + a[4] * T4
    
    # Denominator: 1 + c1*T1 + c2*T2 + c3*T3 + c4*T4
    # Use params[5:9] for c1..c4, and scale params[9] as a global scale for denominator perturbation
    c = params[5:9]
    scale = np.tanh(params[9])  # Scale in [-1, 1] to keep denominator well-behaved
    
    denominator = 1.0 + scale * (c[0] * T1 + c[1] * T2 + c[2] * T3 + c[3] * T4)
    
    # Ensure denominator doesn't get too close to zero
    eps = 1e-8
    sign = np.where(denominator >= 0, 1.0, -1.0)
    safe_denominator = np.where(np.abs(denominator) < eps, eps * sign, denominator)
    
    result = numerator / safe_denominator
    
    # Clip to avoid extreme values
    result = np.clip(result, -1e4, 1e4)
    
    return result.reshape(original_shape)
# EVOLVE-BLOCK-END
