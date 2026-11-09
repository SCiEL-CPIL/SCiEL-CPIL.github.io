"""Symbolic regression for a nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Parametric limit-cycle oscillator structure:
    a = -k(x) * x - c(v) * v + F(t)
    
    where k(x) and c(v) are smooth bounded functions using tanh,
    and F(t) is a sinusoidal driving force.
    
    Structure:
    k(x) = p0 + p1 * tanh(p2 * x)
    c(v) = p3 + p4 * tanh(p5 * v)
    F(t) = p6 * sin(p7 * t) + p8 * cos(p7 * t) + p9
    
    params: array of 10 fitted constants
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Smooth bounded stiffness function
    k_x = p0 + p1 * np.tanh(p2 * x)
    
    # Smooth bounded damping function
    c_v = p3 + p4 * np.tanh(p5 * v)
    
    # Sinusoidal driving force
    F_t = p6 * np.sin(p7 * t) + p8 * np.cos(p7 * t) + p9
    
    # Acceleration
    result = -k_x * x - c_v * v + F_t
    
    return result
# EVOLVE-BLOCK-END
