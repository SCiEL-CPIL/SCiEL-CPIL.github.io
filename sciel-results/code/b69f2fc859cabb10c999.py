"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Autonomous self-sustained oscillator with limit cycle behavior.
    
    Model: a = p1*x + p2*x^3 + p3*v + p4*v^3 + p5*x^2 + p6*v^2 + p7*x*v + p8*x^2*v + p9*v*x^2 + p10
    
    This combines:
    - Duffing-type position nonlinearity: p1*x + p2*x^3
    - Van der Pol-style velocity terms: p3*v + p4*v^3
    - Quadratic position: p5*x^2
    - Quadratic velocity: p6*v^2
    - Cross term: p7*x*v
    - Higher-order coupling: p8*x^2*v, p9*v*x^2
    - Constant offset: p10
    
    Note: Explicit time-dependent driving terms are excluded to focus on intrinsic dynamics.
    """
    # Extract parameters
    p1 = params[0]
    p2 = params[1]
    p3 = params[2]
    p4 = params[3]
    p5 = params[4]
    p6 = params[5]
    p7 = params[6]
    p8 = params[7]
    p9 = params[8]
    p10 = params[9]
    
    # Compute powers
    x2 = x * x
    v2 = v * v
    
    # Duffing potential terms
    duffing = p1 * x + p2 * x * x2
    
    # Van der Pol damping terms (linear + cubic)
    damping = p3 * v + p4 * v * v2
    
    # Quadratic position nonlinearity
    quad_pos = p5 * x2
    
    # Quadratic velocity term
    quad_vel = p6 * v2
    
    # Cross coupling term
    cross = p7 * x * v
    
    # Higher order couplings
    high_coupling = p8 * x2 * v + p9 * v * x2  # Simplified: p8*x^2*v + p9*x^2*v = (p8+p9)*x^2*v
    # Actually, let's keep them distinct in case the optimizer wants to separate them, though they are algebraically identical if both are x^2*v.
    # Let's use distinct terms to allow more flexibility if I mis-specified the symmetry.
    # Term 8: x^2 * v
    # Term 9: x * v^2
    term8 = p8 * x2 * v
    term9 = p9 * x * v2
    
    # Constant bias
    bias = p10
    
    # Total acceleration
    a = duffing + damping + quad_pos + quad_vel + cross + term8 + term9 + bias
    
    return a
# EVOLVE-BLOCK-END
