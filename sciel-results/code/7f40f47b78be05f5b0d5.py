"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale input to [-1, 1] for better numerical stability of Chebyshev polynomials
    u = x / 2.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(u) = 1
    # T_1(u) = u
    # T_n(u) = 2u T_{n-1}(u) - T_{n-2}(u)
    
    T0 = np.ones_like(u)
    T1 = u
    
    # Precompute up to T_9
    T = [T0, T1]
    for n in range(2, 10):
        Tn = 2.0 * u * T[n-1] - T[n-2]
        T.append(Tn)
    
    # Linear combination
    y = params[0] * T0 + params[1] * T1 + params[2] * T[2] + params[3] * T[3] + \
        params[4] * T[4] + params[5] * T[5] + params[6] * T[6] + params[7] * T[7] + \
        params[8] * T[8] + params[9] * T[9]
        
    return y
# EVOLVE-BLOCK-END
