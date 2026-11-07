"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid Chebyshev-Fourier basis with 10 parameters.
    
    We use a combination of:
    - Chebyshev polynomials T_0 through T_4 (5 terms) mapped to [-1,1]
    - Sinusoidal terms sin(x), cos(x), sin(2x), cos(2x), sin(3x) (5 terms)
    
    This hybrid approach captures both smooth polynomial-like behavior (via Chebyshev)
    and oscillatory features (via Fourier terms), providing a more expressive
    and well-conditioned 10-parameter linear model than either pure basis alone.
    
    The Chebyshev polynomials are stable on [-1,1] and provide good approximation
    for smooth functions. The sinusoidal terms capture periodic or oscillatory
    components that polynomials struggle with.
    
    Basis functions:
    1. T_0(t) = 1
    2. T_1(t) = t
    3. T_2(t) = 2t^2 - 1
    4. T_3(t) = 4t^3 - 3t
    5. T_4(t) = 8t^4 - 8t^2 + 1
    6. sin(x)
    7. cos(x)
    8. sin(2x)
    9. cos(2x)
    10. sin(3x)
    """
    # Map x from [-2, 2] to t in [-1, 1] for Chebyshev stability
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 through T_4
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    
    # Sinusoidal basis
    S1 = np.sin(x)
    C1 = np.cos(x)
    S2 = np.sin(2.0 * x)
    C2 = np.cos(2.0 * x)
    S3 = np.sin(3.0 * x)
    
    # Linear combination
    result = (params[0] * T0 +
              params[1] * T1 +
              params[2] * T2 +
              params[3] * T3 +
              params[4] * T4 +
              params[5] * S1 +
              params[6] * C1 +
              params[7] * S2 +
              params[8] * C2 +
              params[9] * S3)
    
    return result
# EVOLVE-BLOCK-END
