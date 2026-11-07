"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for a nonlinear oscillator.
    
    Model:
    y = p0 * t + p1 * x + p2 * v + p3
    
    Parameters:
    p0: Linear time coefficient
    p1: Linear position coefficient
    p2: Linear velocity coefficient
    p3: Constant offset
    """
    p0, p1, p2, p3 = params[:4]
    
    return p0 * t + p1 * x + p2 * v + p3
# EVOLVE-BLOCK-END
