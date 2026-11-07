import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator modeled as a full cubic polynomial
    in scaled position and velocity.
    """
    # Scale inputs to improve numerical conditioning for the optimizer.
    # Based on training ranges: x in [-0.57, 0.53], v in [-0.18, 0.28]
    # Using 0.5 and 0.3 as characteristic scales.
    X = x / 0.5
    V = v / 0.3
    
    # Precompute powers and products
    X2 = X * X
    X3 = X2 * X
    V2 = V * V
    V3 = V2 * V
    XV = X * V
    X2V = X2 * V
    XV2 = X * V2
    
    # Construct the polynomial expression
    # p0: constant
    # p1: X
    # p2: V
    # p3: X^2
    # p4: X*V
    # p5: V^2
    # p6: X^3
    # p7: X^2*V
    # p8: X*V^2
    # p9: V^3
    
    a = (params[0] 
         + params[1] * X 
         + params[2] * V 
         + params[3] * X2 
         + params[4] * XV 
         + params[5] * V2 
         + params[6] * X3 
         + params[7] * X2V 
         + params[8] * XV2 
         + params[9] * V3)
         
    return a
