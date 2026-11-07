"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Autonomous nonlinear oscillator with limit cycle behavior.
    
    Model: a = p1*x + p2*x^3 + p3*v + p4*v^3 + p5*x*v + p6*x^2*v + p7*v^2 + p8*x^2 + p9*x*v^2 + p10
    
    This model excludes explicit time-dependent driving terms, relying on the
    intrinsic dynamics of the autonomous system to fit the data.
    
    Terms:
    - Duffing restoring force: p1*x + p2*x^3
    - Van der Pol damping: p3*v + p4*v^3
    - Cross-coupling and higher-order nonlinearities: p5*x*v, p6*x^2*v, p7*v^2, p8*x^2, p9*x*v^2
    - Constant offset: p10
    """
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = params
    
    x2 = x * x
    v2 = v * v
    
    # Duffing potential terms
    restoring = p1 * x + p2 * x * x2
    
    # Van der Pol style damping (linear and cubic)
    damping = p3 * v + p4 * v * v2
    
    # Additional nonlinear cross terms to capture complex dynamics
    cross_terms = p5 * x * v + p6 * x2 * v + p7 * v2 + p8 * x2 + p9 * x * v2
    
    # Constant bias
    bias = p10
    
    # Total acceleration
    a = restoring + damping + cross_terms + bias
    
    return a
# EVOLVE-BLOCK-END
