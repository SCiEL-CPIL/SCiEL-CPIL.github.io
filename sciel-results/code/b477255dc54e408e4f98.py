"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Neural Network-like model for nonlinear oscillator.
    
    Uses a sum of two tanh-activated units to capture non-linear interactions
    between time, position, and velocity, plus a linear residual term.
    
    Structure:
    y = p0 * tanh(p1*t + p2*x + p3*v) + p4 * tanh(p5*t + p6*x + p7*v) + p8*x + p9*v
    """
    p = params
    
    # Two hidden units with tanh activation
    h1 = np.tanh(p[1] * t + p[2] * x + p[3] * v)
    h2 = np.tanh(p[5] * t + p[6] * x + p[7] * v)
    
    # Linear combination of hidden units and linear terms
    return p[0] * h1 + p[4] * h2 + p[8] * x + p[9] * v
# EVOLVE-BLOCK-END
