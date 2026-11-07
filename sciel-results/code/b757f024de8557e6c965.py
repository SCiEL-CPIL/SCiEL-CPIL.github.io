"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    Uses a basis motivated by Van der Pol and Duffing oscillators.
    Inputs are scaled to improve numerical conditioning.
    """
    # Scale inputs to approximately [-1, 1] based on provided ranges
    # x range: [-0.5728, 0.5275] -> scale ~ 0.55
    # v range: [-0.1754, 0.2821] -> scale ~ 0.25
    # Using slightly wider scales to be safe
    x_scale = 0.6
    v_scale = 0.3
    
    X = x / x_scale
    V = v / v_scale
    
    # Precompute powers
    X2 = X * X
    X3 = X2 * X
    X4 = X2 * X2
    V2 = V * V
    V3 = V2 * V
    
    # Cross terms
    XV = X * V
    X2V = X2 * V
    X3V = X3 * V
    
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Basis functions:
    # p0: bias
    # p1: X (linear restoring)
    # p2: V (linear damping)
    # p3: X^3 (cubic stiffness, Duffing)
    # p4: X^2 * V (Van der Pol damping term: (1-X^2)V = V - X^2 V)
    # p5: V^3 (cubic damping)
    # p6: X * V (cross coupling)
    # p7: X^2 (quadratic asymmetry)
    # p8: V^2 (quadratic drag)
    # p9: X^3 * V (higher order damping correction)
    
    return (p0 + p1 * X + p2 * V + p3 * X3 + p4 * X2V + p5 * V3 + p6 * XV + p7 * X2 + p8 * V2 + p9 * X3V)
# EVOLVE-BLOCK-END
