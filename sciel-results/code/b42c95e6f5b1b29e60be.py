# EVOLVE-BLOCK-START
import numpy as np
from scipy import optimize


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Use Chebyshev polynomials with improved conditioning.
    # Scale x to [-1, 1]
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    T0 = np.ones_like(t)
    T1 = t.copy()
    
    T2 = 2 * t * T1 - T0
    T3 = 2 * t * T2 - T1
    T4 = 2 * t * T3 - T2
    T5 = 2 * t * T4 - T3
    T6 = 2 * t * T5 - T4
    T7 = 2 * t * T6 - T5
    T8 = 2 * t * T7 - T6
    T9 = 2 * t * T8 - T7
    
    # Apply gentle decay to higher-order terms to improve conditioning
    # and help BFGS converge from the all-ones initialization
    scale = np.array([1.0, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15])
    
    # Linear combination with scaled parameters
    y = (scale[0] * params[0] * T0 + 
         scale[1] * params[1] * T1 + 
         scale[2] * params[2] * T2 + 
         scale[3] * params[3] * T3 + 
         scale[4] * params[4] * T4 + 
         scale[5] * params[5] * T5 + 
         scale[6] * params[6] * T6 + 
         scale[7] * params[7] * T7 + 
         scale[8] * params[8] * T8 + 
         scale[9] * params[9] * T9)
         
    return y
# EVOLVE-BLOCK-END
