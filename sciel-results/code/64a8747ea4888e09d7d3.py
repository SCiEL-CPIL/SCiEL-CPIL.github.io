import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Van der Pol-inspired nonlinear oscillator model.
    
    Uses a basis that explicitly captures amplitude-dependent damping (limit cycle dynamics)
    and higher-order nonlinearities, combined with input scaling for numerical stability.
    
    Model structure:
    a = p0 + p1*x + p2*v + p3*x^3 + p4*v^2 + p5*xv + p6*x^5 + p7*v^3 + p8*x^2*v + p9*x^4
    
    Inputs are scaled to [-1, 1] based on training ranges to improve conditioning.
    """
    p = params
    
    # Scale inputs based on training ranges to improve conditioning
    # x range: [-0.5728, 0.5275]
    # v range: [-0.1754, 0.2821]
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
    X5 = X4 * X
    
    V2 = V * V
    V3 = V2 * V
    
    XV = X * V
    X2V = X2 * V
    
    # Basis terms:
    # 0: Constant
    # 1: Linear X (restoring)
    # 2: Linear V (damping)
    # 3: Cubic X (Duffing nonlinearity)
    # 4: Quadratic V (amplitude-dependent damping, Van der Pol)
    # 5: Cross term X*V (coupling)
    # 6: Quintic X (higher-order nonlinearity)
    # 7: Cubic V (nonlinear damping)
    # 8: Cross term X^2*V
    # 9: Quartic X (even nonlinearity)
    
    return (p[0] + 
            p[1] * X + 
            p[2] * V + 
            p[3] * X3 + 
            p[4] * V2 + 
            p[5] * XV + 
            p[6] * X5 + 
            p[7] * V3 + 
            p[8] * X2V + 
            p[9] * X4)
