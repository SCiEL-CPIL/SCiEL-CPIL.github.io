import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a physics-informed sparse basis.
    
    Inputs:
    x: Position
    v: Velocity
    params: Array of 10 fitted constants
    
    Returns:
    Acceleration
    """
    # Scale inputs for better conditioning
    # x range ~ [-0.57, 0.53], center ~0, scale ~0.55
    # v range ~ [-0.18, 0.28], center ~0.05, scale ~0.25
    X = x / 0.55
    V = (v - 0.05) / 0.25
    
    # Precompute powers
    X2 = X * X
    X3 = X2 * X
    V2 = V * V
    V3 = V2 * V
    XV = X * V
    
    # Physics-informed terms for nonlinear oscillator:
    # p0: Linear stiffness (Hooke's law)
    term0 = X
    # p1: Cubic stiffness (Duffing nonlinearity)
    term1 = X3
    # p2: Linear damping
    term2 = V
    # p3: Quadratic damping
    term3 = V2
    # p4: Cubic damping
    term4 = V3
    # p5: Velocity-position coupling
    term5 = XV
    # p6: Position-squared times velocity
    term6 = X2 * V
    # p7: Position times velocity-squared
    term7 = X * V2
    # p8: Higher-order position nonlinearity
    term8 = X2 * V2
    # p9: Constant offset
    term9 = np.ones_like(x)
    
    # Combine with parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    return (p0 * term0 + 
            p1 * term1 + 
            p2 * term2 + 
            p3 * term3 + 
            p4 * term4 + 
            p5 * term5 + 
            p6 * term6 + 
            p7 * term7 + 
            p8 * term8 + 
            p9 * term9)
