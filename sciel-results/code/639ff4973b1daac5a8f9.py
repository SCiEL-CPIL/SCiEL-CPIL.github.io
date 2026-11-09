"""Physics-informed Duffing-van der Pol oscillator model with structured nonlinearities."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with driving force, from position and velocity.
    
    Uses a Duffing-van der Pol structure:
    a = p1*x + p2*x^3 - p3*v - p4*v^3 + p5*x*v^2 + p6*v^2*x^2 + p7*x^2*v + p8*x^4 + p9*v^4 + p10
    
    Parameters:
        x: position
        v: velocity
        params: array of 10 fitted constants
    """
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = params
    
    # Restoring force (Duffing): linear + cubic + quartic
    restoring = p1 * x + p2 * x3 + p8 * x4
    
    # Damping terms: linear, cubic, and higher-order velocity
    damping = -p3 * v - p4 * v3 + p9 * v4
    
    # Nonlinear coupling terms
    coupling = p5 * x * v2 + p6 * v2 * x2 + p7 * x2 * v
    
    # Constant offset
    offset = p10
    
    return restoring + damping + coupling + offset
# EVOLVE-BLOCK-END
