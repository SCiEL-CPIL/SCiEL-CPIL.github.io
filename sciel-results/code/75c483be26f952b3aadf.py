"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 150.0
_TIME_MARGIN = 5.0  # safety margin in seconds


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model with linear, interaction, and sinusoidal non-linear terms.
    
    The previous centered quadratic model scored poorly (7.39), suggesting that
    either the function is not well-approximated by a quadratic, or the optimizer
    struggled with the specific basis. A linear model scored 1.0, indicating
    significant non-linearity.
    
    Strategy:
    - Use a basis that includes interactions (xy, xz, yz) and simple non-linear
      functions (sin) which are common in symbolic regression targets.
    - This basis is generally better conditioned than high-order polynomials
      and avoids the potential pitfalls of exponential terms blowing up or
      being too flat.
    
    Basis allocation (10 params):
    p0: bias (1)
    p1: x
    p2: y
    p3: z
    p4: x*y
    p5: x*z
    p6: y*z
    p7: sin(x)
    p8: sin(y)
    p9: sin(z)
    """
    # Check time budget
    if time.time() - _START_TIME > _TIME_LIMIT - _TIME_MARGIN:
        # Return a simple linear prediction as fallback
        return params[0] + params[1] * x + params[2] * y + params[3] * z
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    
    p = params
    
    # Build the prediction
    pred = (p[0] * np.ones_like(x) +
            p[1] * x +
            p[2] * y +
            p[3] * z +
            p[4] * x * y +
            p[5] * x * z +
            p[6] * y * z +
            p[7] * np.sin(x) +
            p[8] * np.sin(y) +
            p[9] * np.sin(z))
    
    return pred
# EVOLVE-BLOCK-END
