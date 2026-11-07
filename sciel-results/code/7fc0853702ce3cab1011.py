"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 110.0  # 110s safety margin within 120s limit


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Product-based feature expansion: Chebyshev polynomials multiplied by
    nonlinear activations (tanh) to capture non-periodic nonlinearities.
    
    Structure:
    - 5 terms: T_k(t) * tanh(c_k * t) for k=0..4
    - 5 terms: T_k(t) * sin(omega_k * t) for k=0..4
    
    This hybrid basis combines global polynomial trends with localized
    nonlinear features, providing a flexible yet stable representation
    for BFGS optimization.
    """
    if time.time() > _DEADLINE:
        # Fallback: return a simple linear fit as safety
        x = np.asarray(x, dtype=np.float64)
        return params[0] + params[1] * x
    
    x = np.asarray(x, dtype=np.float64)
    
    # Normalize x to [-1, 1] for Chebyshev stability
    t = x / 3.0
    
    # Chebyshev Polynomials T_0 to T_4
    b0 = np.ones_like(x)
    b1 = t
    b2 = 2.0 * t * t - 1.0
    b3 = 4.0 * b1 * b1 * b1 - 3.0 * b1
    b4 = 8.0 * b1 * b1 * b1 * b1 - 8.0 * b1 * b1 + 1.0
    
    # Nonlinear activations: tanh with different scales
    tanh1 = np.tanh(1.0 * t)
    tanh2 = np.tanh(2.0 * t)
    tanh3 = np.tanh(0.5 * t)
    tanh4 = np.tanh(3.0 * t)
    tanh5 = np.tanh(1.5 * t)
    
    # Nonlinear activations: sine with different frequencies
    sin1 = np.sin(1.0 * t)
    sin2 = np.sin(2.0 * t)
    sin3 = np.sin(0.5 * t)
    sin4 = np.sin(3.0 * t)
    sin5 = np.sin(1.5 * t)
    
    # Product features: Chebyshev * tanh
    f0 = b0 * tanh1
    f1 = b1 * tanh2
    f2 = b2 * tanh3
    f3 = b3 * tanh4
    f4 = b4 * tanh5
    
    # Product features: Chebyshev * sin
    f5 = b0 * sin1
    f6 = b1 * sin2
    f7 = b2 * sin3
    f8 = b3 * sin4
    f9 = b4 * sin5
    
    # Linear combination
    y = (params[0] * f0 + 
         params[1] * f1 + 
         params[2] * f2 + 
         params[3] * f3 + 
         params[4] * f4 + 
         params[5] * f5 + 
         params[6] * f6 + 
         params[7] * f7 + 
         params[8] * f8 + 
         params[9] * f9)
         
    return y
# EVOLVE-BLOCK-END
