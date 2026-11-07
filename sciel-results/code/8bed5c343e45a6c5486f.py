import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Van der Pol oscillator model with amplitude-dependent damping.
    
    The Van der Pol oscillator equation is:
    x'' = mu*(1 - x^2)*x' - x
    
    This gives acceleration as a function of position x and velocity v (where v = x').
    
    Model: a = -x + mu*(1 - x^2)*v
    
    Expanded: a = -x + mu*v - mu*x^2*v
    
    We use a richer basis to capture potential variations:
    - Linear restoring: x
    - Linear damping: v
    - Van der Pol damping: x^2 * v
    - Higher-order terms for flexibility
    
    Basis terms (10 total):
    0: constant
    1: x (linear restoring)
    2: v (linear damping)
    3: x^2 * v (Van der Pol nonlinear damping)
    4: x^3 (cubic stiffness)
    5: v^2 (quadratic velocity term)
    6: x * v (cross term)
    7: x^4 * v (higher-order Van der Pol)
    8: v^3 (cubic velocity)
    9: x^2 (quadratic position)
    
    We scale inputs for numerical stability.
    """
    p = params
    
    # Scale inputs based on training ranges
    # x range: [-0.5728, 0.5275], center ≈ -0.02265, scale ≈ 0.55015
    # v range: [-0.1754, 0.2821], center ≈ 0.05335, scale ≈ 0.22875
    x_center = -0.02265
    x_scale = 0.55015
    v_center = 0.05335
    v_scale = 0.22875
    
    X = (x - x_center) / x_scale
    V = (v - v_center) / v_scale
    
    # Pre-compute monomials
    X2 = X * X
    X3 = X2 * X
    X4 = X2 * X2
    
    V2 = V * V
    V3 = V2 * V
    
    XV = X * V
    X2V = X2 * V
    X4V = X4 * V
    
    return (p[0] +
            p[1] * X +
            p[2] * V +
            p[3] * X2V +
            p[4] * X3 +
            p[5] * V2 +
            p[6] * XV +
            p[7] * X4V +
            p[8] * V3 +
            p[9] * X2)
