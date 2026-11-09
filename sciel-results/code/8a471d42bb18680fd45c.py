"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model using a single hidden-layer neuron (tanh activation) 
    combined with linear terms and an interaction term.
    
    Structure:
    y = p4 * tanh(p0*xn + p1*vn + p2*tn + p3) 
      + p5*xn + p6*vn + p7*tn + p8*xn*vn + p9
    
    where xn, vn, tn are normalized inputs to O(1) scale.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to improve conditioning for BFGS
    # Ranges from prompt: t[30,50], x[-0.2056, 0.1749], v[-0.3391, 0.3624]
    # Center t at 40, scale by 10 (half-range)
    # Center x at 0, scale by 0.21 (max abs approx)
    # Center v at 0, scale by 0.36 (max abs approx)
    
    x_norm = x / 0.21
    v_norm = v / 0.36
    t_norm = (t - 40.0) / 10.0
    
    # Nonlinear term: Single tanh neuron
    # Inputs to neuron are linear combination of normalized features
    z = p0 * x_norm + p1 * v_norm + p2 * t_norm + p3
    nonlin_part = p4 * np.tanh(z)
    
    # Linear terms and interaction
    linear_part = p5 * x_norm + p6 * v_norm + p7 * t_norm
    interaction_part = p8 * x_norm * v_norm
    
    return nonlin_part + linear_part + interaction_part + p9
# EVOLVE-BLOCK-END
