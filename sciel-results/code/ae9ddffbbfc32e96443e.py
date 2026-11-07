"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for a nonlinear oscillator.
    
    Model:
    y = p0 * cos(p1 * t + p2) + p3 * x + p4 * x^3 + p5 * v 
        + p6 * x^2 + p7 * v^2 + p8 * x * v + p9
    
    Parameters:
    p0: Amplitude of driving force
    p1: Frequency of driving force
    p2: Phase of driving force
    p3: Linear stiffness coefficient
    p4: Nonlinear (cubic) stiffness coefficient
    p5: Damping coefficient
    p6: Quadratic position term
    p7: Quadratic velocity term
    p8: Velocity-position cross term
    p9: Constant offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Driving term
    drive = p0 * np.cos(p1 * t + p2)
    
    # Position terms
    pos_term = p3 * x + p4 * (x ** 3) + p6 * (x ** 2)
    
    # Velocity terms
    vel_term = p5 * v + p7 * (v ** 2)
    
    # Interaction terms
    cross_term = p8 * x * v
    
    return drive + pos_term + vel_term + cross_term + p9
# EVOLVE-BLOCK-END
