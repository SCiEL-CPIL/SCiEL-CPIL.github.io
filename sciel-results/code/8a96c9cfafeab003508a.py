"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: Chebyshev polynomial basis (degrees 0-5) + bounded sinusoidal basis (6-9).
    
    Design rationale:
    - Chebyshev T_0..T_5 on scaled u in [-1,1]: well-conditioned, captures polynomial trends.
    - sin/cos basis at frequencies 1-3 rad per unit-x: bounded in [-1,1], captures periodic
      structure without the ill-conditioning of high-degree Chebyshev.
    
    This hybrid maintains global approximation capacity while keeping the design matrix
    well-conditioned, helping BFGS converge from the all-ones start to a better optimum.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Scale x from [-3, 3] to [-1, 1]
    u = x / 3.0
    
    # Compute Chebyshev polynomials T_0 through T_5
    T0 = np.ones_like(u)
    T1 = u.copy()
    
    result = params[0] * T0 + params[1] * T1
    
    T_prev2 = T0
    T_prev1 = T1
    for k in range(2, 6):
        T_curr = 2.0 * u * T_prev1 - T_prev2
        result += params[k] * T_curr
        T_prev2 = T_prev1
        T_prev1 = T_curr
    
    # Parameters 6-9: sinusoidal basis functions (bounded, well-conditioned)
    # Use frequencies that provide good coverage of [-3,3]
    # sin(x), cos(x), sin(2x), cos(2x)
    if len(params) > 6:
        result += params[6] * np.sin(x)
    if len(params) > 7:
        result += params[7] * np.cos(x)
    if len(params) > 8:
        result += params[8] * np.sin(2.0 * x)
    if len(params) > 9:
        result += params[9] * np.cos(2.0 * x)
    
    return result
# EVOLVE-BLOCK-END
