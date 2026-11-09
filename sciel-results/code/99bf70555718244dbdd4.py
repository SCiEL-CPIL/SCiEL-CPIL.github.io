"""Physics-informed Duffing-van der Pol oscillator model with structured nonlinearities."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with driving force, from position and velocity.
    
    Uses a Duffing-van der Pol inspired structure with physically motivated terms:
    a = p1*x + p2*x^3 + p3*x^2 + p4*v + p5*v^3 + p6*x*v + p7*x^2*v + p8*v^2*x + p9*x^4 + p10
    
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
    
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = params
    
    # Core restoring force (Duffing): linear + cubic + quadratic correction
    restoring = p1 * x + p2 * x3 + p3 * x2
    
    # Damping terms: linear, cubic, and coupling with position
    damping = p4 * v + p5 * v3 + p6 * x * v + p7 * x2 * v
    
    # Additional nonlinear coupling
    nonlinear_coupling = p8 * v2 * x
    
    # Higher-order position nonlinearity
    higher_order = p9 * x4
    
    # Constant offset
    offset = p10
    
    return restoring + damping + nonlinear_coupling + higher_order + offset
# EVOLVE-BLOCK-END
