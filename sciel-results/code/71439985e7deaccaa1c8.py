"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x to [-1, 1] for Chebyshev stability
    t = x / 3.0
    
    # Chebyshev polynomials T_0 to T_7 (higher order for better approximation)
    # T_0 = 1
    # T_1 = t
    # T_2 = 2t^2 - 1
    # T_3 = 4t^3 - 3t
    # T_4 = 8t^4 - 8t^2 + 1
    # T_5 = 16t^5 - 20t^3 + 5t
    # T_6 = 32t^6 - 48t^4 + 18t^2 - 1
    # T_7 = 64t^7 - 112t^5 + 56t^3 - 7t
    
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t**2 - 1.0
    T3 = 4.0 * t**3 - 3.0 * t
    T4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    T5 = 16.0 * t**5 - 20.0 * t**3 + 5.0 * t
    T6 = 32.0 * t**6 - 48.0 * t**4 + 18.0 * t**2 - 1.0
    T7 = 64.0 * t**7 - 112.0 * t**5 + 56.0 * t**3 - 7.0 * t
    
    # Linear combination of Chebyshev terms
    # params[0] to params[7]
    y_cheb = (params[0] * T0 + 
              params[1] * T1 + 
              params[2] * T2 + 
              params[3] * T3 + 
              params[4] * T4 + 
              params[5] * T5 +
              params[6] * T6 +
              params[7] * T7)
    
    # Sinusoidal term: A * sin(w * x)
    # params[8] is amplitude A
    # params[9] is raw frequency, mapped to [-5, 5] via tanh
    A = params[8]
    w = np.tanh(params[9]) * 5.0
    y_sin = A * np.sin(w * x)
    
    return y_cheb + y_sin
# EVOLVE-BLOCK-END
