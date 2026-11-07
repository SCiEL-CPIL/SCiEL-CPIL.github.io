"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator modeled as a full cubic polynomial
    in scaled position and velocity.
    
    The basis includes:
    - Constant term
    - Linear terms in X and V
    - Quadratic terms: X^2, XV, V^2
    - Cubic terms: X^3, X^2V, XV^2, V^3
    - Additional cross terms for flexibility: X^2V^2, X^3V, XV^3
    
    Inputs are scaled to O(1) range to improve numerical conditioning for BFGS.
    """
    # Scale inputs based on training ranges:
    # x in [-0.5728, 0.5275] -> scale by 0.55
    # v in [-0.1754, 0.2821] -> scale by 0.25
    X = x / 0.55
    V = v / 0.25
    
    # Precompute powers and products
    X2 = X * X
    X3 = X2 * X
    V2 = V * V
    V3 = V2 * V
    XV = X * V
    X2V = X2 * V
    XV2 = X * V2
    X2V2 = X2 * V2
    
    # Construct the polynomial expression with 10 parameters
    # p0: constant
    # p1: X (linear position)
    # p2: V (linear velocity)
    # p3: X^2 (quadratic position)
    # p4: XV (cross term)
    # p5: V^2 (quadratic velocity)
    # p6: X^3 (cubic position - Duffing-like)
    # p7: X^2V (cubic cross)
    # p8: XV^2 (cubic cross)
    # p9: V^3 (cubic velocity)
    
    result = (
        params[0] +
        params[1] * X +
        params[2] * V +
        params[3] * X2 +
        params[4] * XV +
        params[5] * V2 +
        params[6] * X3 +
        params[7] * X2V +
        params[8] * XV2 +
        params[9] * V3
    )
    
    return result
# EVOLVE-BLOCK-END
