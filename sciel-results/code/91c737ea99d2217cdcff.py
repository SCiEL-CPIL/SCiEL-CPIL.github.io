# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a damped nonlinear oscillator with driving force, from position and velocity."""
    x2 = x * x
    v2 = v * v
    x3 = x2 * x
    v3 = v2 * v
    x2v = x2 * v
    xv2 = x * v2
    
    # Sparse basis: likely dominant terms for nonlinear oscillators
    # 1. x
    # 2. x^3
    # 3. v
    # 4. v^3
    # 5. x^2 v
    # 6. x v^2
    # 7. bias
    # We use params[0] to params[6]. params[7..9] are unused.
    
    return (params[0] * x +
            params[1] * x3 +
            params[2] * v +
            params[3] * v3 +
            params[4] * x2v +
            params[5] * xv2 +
            params[6])
# EVOLVE-BLOCK-END
