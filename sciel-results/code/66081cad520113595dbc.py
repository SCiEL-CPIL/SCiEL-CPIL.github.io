import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    Uses a sparse polynomial basis in scaled x and v.
    """
    # Scale inputs to O(1) range to improve conditioning
    # x range approx [-0.57, 0.53] -> scale by 0.5
    # v range approx [-0.18, 0.28] -> scale by 0.2
    X = x / 0.5
    V = v / 0.2

    # Precompute powers
    X2 = X * X
    X3 = X2 * X
    V2 = V * V
    V3 = V2 * V
    XV = X * V
    X2V = X2 * V
    XV2 = X * V2

    # Linear combination of basis functions
    # p0: constant
    # p1: X
    # p2: X^2
    # p3: X^3
    # p4: V
    # p5: V^2
    # p6: X*V
    # p7: X^2*V
    # p8: X*V^2
    # p9: V^3
    
    pred = (params[0] + 
            params[1] * X + 
            params[2] * X2 + 
            params[3] * X3 + 
            params[4] * V + 
            params[5] * V2 + 
            params[6] * XV + 
            params[7] * X2V + 
            params[8] * XV2 + 
            params[9] * V3)
            
    return pred
