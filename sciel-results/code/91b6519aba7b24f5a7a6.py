import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to O(1)
    S_x = 0.5728
    S_v = 0.2821
    X = x / S_x
    V = v / S_v
    
    # Duffing-van der Pol inspired structure with 10 parameters
    # a = p1*X + p2*X^3 - p3*V - p4*V^3 + p5*X*V^2 + p6*V^2*X^2 + p7*X^2*V + p8*X^4 + p9*V^4 + p10
    # This captures the essential nonlinear dynamics of a Duffing oscillator
    # with damping and forcing terms
    
    X2 = X * X
    X3 = X2 * X
    X4 = X2 * X2
    
    V2 = V * V
    V3 = V2 * V
    V4 = V2 * V2
    
    return (params[0] * X + 
            params[1] * X3 - 
            params[2] * V - 
            params[3] * V3 + 
            params[4] * X * V2 + 
            params[5] * V2 * X2 + 
            params[6] * X2 * V + 
            params[7] * X4 + 
            params[8] * V4 + 
            params[9])
