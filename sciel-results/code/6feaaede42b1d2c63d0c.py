"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using Chebyshev polynomials of the first kind.
    
    Maps inputs from [0.5, 2] to [-1, 1] for Chebyshev evaluation,
    then constructs a linear combination of Chebyshev basis terms.
    
    With 10 parameters, we use:
    - Constant term (T0 for all)
    - Linear terms (T1) for x, y, z
    - Quadratic cross terms (T1(x)*T1(y), T1(x)*T1(z), T1(y)*T1(z))
    - Pure quadratic terms (T2(x), T2(y), T2(z))
    
    This gives 1 + 3 + 3 + 3 = 10 parameters total.
    """
    # Map from [0.5, 2] to [-1, 1]
    # t = (2 * (val - 0.5)) / (2 - 0.5) - 1 = (2 * (val - 0.5)) / 1.5 - 1
    # Simplify: t = (4/3) * (val - 0.5) - 1 = (4/3)*val - 2/3 - 1 = (4/3)*val - 5/3
    # Or: t = 2*(val - 1.25)/0.75 - 1 = (2/0.75)*(val - 1.25) - 1 = (8/3)*(val - 1.25) - 1
    # Let's use: center = (0.5 + 2)/2 = 1.25, half_range = (2 - 0.5)/2 = 0.75
    # t = (val - center) / half_range = (val - 1.25) / 0.75
    
    center = 1.25
    half_range = 0.75
    
    tx = (x - center) / half_range
    ty = (y - center) / half_range
    tz = (z - center) / half_range
    
    # Clip to [-1, 1] to avoid numerical issues (shouldn't be needed for well-formed inputs)
    tx = np.clip(tx, -1.0, 1.0)
    ty = np.clip(ty, -1.0, 1.0)
    tz = np.clip(tz, -1.0, 1.0)
    
    # Chebyshev polynomials of the first kind
    # T0(t) = 1
    # T1(t) = t
    # T2(t) = 2*t^2 - 1
    # T3(t) = 4*t^3 - 3*t
    
    T0 = np.ones_like(tx)
    T1x = tx
    T1y = ty
    T1z = tz
    T2x = 2.0 * tx**2 - 1.0
    T2y = 2.0 * ty**2 - 1.0
    T2z = 2.0 * tz**2 - 1.0
    T3x = 4.0 * tx**3 - 3.0 * tx
    T3y = 4.0 * ty**3 - 3.0 * ty
    T3z = 4.0 * tz**3 - 3.0 * tz
    
    # Build the linear combination with 10 parameters
    # This basis assumes a separable structure up to degree 3.
    # params[0]: constant
    # params[1]: T1(x)
    # params[2]: T2(x)
    # params[3]: T3(x)
    # params[4]: T1(y)
    # params[5]: T2(y)
    # params[6]: T3(y)
    # params[7]: T1(z)
    # params[8]: T2(z)
    # params[9]: T3(z)
    
    p = params
    
    result = (
        p[0] * T0 +
        p[1] * T1x +
        p[2] * T2x +
        p[3] * T3x +
        p[4] * T1y +
        p[5] * T2y +
        p[6] * T3y +
        p[7] * T1z +
        p[8] * T2z +
        p[9] * T3z
    )
    
    return result
# EVOLVE-BLOCK-END
