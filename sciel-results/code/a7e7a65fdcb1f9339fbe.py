"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a combination of polynomial, tanh-based
    adaptive features, and trigonometric basis functions.
    
    The design uses:
    - A cubic polynomial for global trend
    - Two tanh units with learnable centers/scales for local features
    - Sine/cosine for periodic components
    
    All 10 parameters are used. The expression is smooth and differentiable
    for BFGS optimization.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Polynomial basis: captures global polynomial trend
    poly = p0 * x + p1 * x**2 + p2 * x**3
    
    # Two tanh-based adaptive units: can model bumps, steps, or saturating behaviors
    # Each tanh unit has a center (p4, p7), a scale (p5, p8), and an amplitude (p3, p6)
    # Wait, let me reassign to use all 10 params cleanly:
    # p0: linear coeff
    # p1: quadratic coeff  
    # p2: cubic coeff
    # p3: tanh1 amplitude
    # p4: tanh1 center
    # p5: tanh1 scale (inverse width)
    # p6: tanh2 amplitude
    # p7: tanh2 center
    # p8: tanh2 scale (inverse width)
    # p9: constant offset
    
    # Recompute with clear assignment
    poly = p0 * x + p1 * x**2 + p2 * x**3 + p9
    
    # Tanh unit 1: p3 * tanh(p5 * (x - p4))
    # Clip the argument to avoid numerical issues
    arg1 = np.clip(p5 * (x - p4), -10, 10)
    tanh1 = p3 * np.tanh(arg1)
    
    # Tanh unit 2: p6 * tanh(p8 * (x - p7))
    arg2 = np.clip(p8 * (x - p7), -10, 10)
    tanh2 = p6 * np.tanh(arg2)
    
    return poly + tanh1 + tanh2
# EVOLVE-BLOCK-END
