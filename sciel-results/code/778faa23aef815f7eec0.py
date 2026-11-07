"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator modeled with a symmetric basis
    combining polynomial and trigonometric/hyperbolic functions.
    
    Assumes symmetry f(-x, -v) = -f(x, v), typical for centered oscillators.
    Includes linear, cubic, 5th-order polynomial terms, and saturation terms
    (sin, tanh) to capture nonlinear restoring and damping forces.
    """
    # Scale inputs to improve numerical conditioning for the optimizer.
    # Based on training ranges: x in [-0.57, 0.53], v in [-0.18, 0.28]
    # Using 0.5 and 0.3 as characteristic scales.
    X = x / 0.5
    V = v / 0.3
    
    # Precompute powers
    X2 = X * X
    X3 = X2 * X
    X5 = X3 * X2
    V2 = V * V
    V3 = V2 * V
    V5 = V3 * V2
    
    # Precompute cross terms (odd symmetry)
    X2V = X2 * V
    XV2 = X * V2
    
    # Precompute non-polynomial terms (odd symmetry)
    sinX = np.sin(X)
    tanhV = np.tanh(V)
    
    # Construct the expression
    # p0: X (Linear restoring)
    # p1: V (Linear damping)
    # p2: X^3 (Cubic stiffness)
    # p3: V^3 (Cubic damping)
    # p4: X^2 V (Cross term)
    # p5: X V^2 (Cross term)
    # p6: sin(X) (Nonlinear restoring, bounded)
    # p7: tanh(V) (Nonlinear damping, bounded)
    # p8: X^5 (5th order stiffness)
    # p9: V^5 (5th order damping)
    
    return (params[0] * X + 
            params[1] * V + 
            params[2] * X3 + 
            params[3] * V3 + 
            params[4] * X2V + 
            params[5] * XV2 + 
            params[6] * sinX + 
            params[7] * tanhV + 
            params[8] * X5 + 
            params[9] * V5)
# EVOLVE-BLOCK-END
