"""Symbolic regression for a nonlinear oscillator using a 3rd-order polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator.
    Uses a full 3rd-order polynomial in x and v to capture nonlinear restoring forces,
    damping, and coupling terms.
    
    Args:
        x: Position array.
        v: Velocity array.
        params: Array of 10 fitted constants.
        
    Returns:
        Acceleration array.
    """
    # Precompute powers and cross terms for efficiency
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # Basis terms:
    # 0: constant (driving force bias)
    # 1: x (linear restoring)
    # 2: v (linear damping)
    # 3: x^2 (quadratic nonlinearity)
    # 4: x*v (coupling)
    # 5: v^2 (quadratic damping/drag)
    # 6: x^3 (cubic restoring, e.g. Duffing)
    # 7: x^2*v (cubic damping)
    # 8: x*v^2 (cubic coupling)
    # 9: v^3 (cubic damping)
    
    return (params[0] * 1.0 + 
            params[1] * x + 
            params[2] * v + 
            params[3] * x2 + 
            params[4] * x * v + 
            params[5] * v2 + 
            params[6] * x3 + 
            params[7] * x2 * v + 
            params[8] * x * v2 + 
            params[9] * v3)
# EVOLVE-BLOCK-END
