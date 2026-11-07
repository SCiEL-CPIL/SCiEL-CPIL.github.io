"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: The evaluator kills after 150s wall-clock.
_START_TIME = time.time()
_MAX_TIME = 145.0

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis function combining high-order Chebyshev polynomials (normalized)
    and diverse non-polynomial functions (trig, hyperbolic, exponential).
    
    The input x is in [-3, 3]. We normalize x to [-1, 1] for Chebyshev stability.
    We use a mix of 7 Chebyshev polynomials (T_0 to T_6) to capture smooth
    polynomial-like behavior up to degree 6, and 3 diverse non-polynomial
    features (sin, cos, tanh) to capture oscillatory or saturating behaviors.
    
    Basis:
    0: T_0(z) = 1
    1: T_1(z) = z
    2: T_2(z) = 2z^2 - 1
    3: T_3(z) = 4z^3 - 3z
    4: T_4(z) = 8z^4 - 8z^2 + 1
    5: T_5(z) = 16z^5 - 20z^3 + 5z
    6: T_6(z) = 32z^6 - 48z^4 + 18z^2 - 1
    7: sin(x)
    8: cos(x)
    9: tanh(x)
    
    This provides a rich manifold for BFGS to fit various hidden functions.
    Chebyshev polynomials are orthogonal on [-1, 1] and provide stable
    approximations for smooth functions. The trig and tanh terms add
    non-polynomial flexibility.
    """
    # Check time limit
    if time.time() - _START_TIME > _MAX_TIME:
        # Return a safe default if we are about to time out
        return np.zeros_like(x, dtype=np.float64)

    x = np.asarray(x, dtype=np.float64)
    
    # Normalize x for Chebyshev basis
    z = x / 3.0
    
    # Compute Chebyshev polynomials T_0 to T_6 using recurrence
    # T_0(z) = 1
    # T_1(z) = z
    # T_n(z) = 2*z*T_{n-1}(z) - T_{n-2}(z)
    
    T0 = np.ones_like(x)
    T1 = z
    T2 = 2.0 * z * T1 - T0
    T3 = 2.0 * z * T2 - T1
    T4 = 2.0 * z * T3 - T2
    T5 = 2.0 * z * T4 - T3
    T6 = 2.0 * z * T5 - T4
    
    # Non-polynomial features on raw x
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    tanh_x = np.tanh(x)
    
    # Linear combination
    # params is length 10
    pred = (params[0] * T0 + 
            params[1] * T1 + 
            params[2] * T2 + 
            params[3] * T3 + 
            params[4] * T4 + 
            params[5] * T5 + 
            params[6] * T6 + 
            params[7] * sin_x + 
            params[8] * cos_x + 
            params[9] * tanh_x)
            
    return pred
# EVOLVE-BLOCK-END
