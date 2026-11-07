import numpy as np

# EVOLVE-BLOCK-START

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model (Duffing-like with forcing).
    
    Model:
    a = p0 + p1*X + p2*V + p3*X^2 + p4*V^2 + p5*X*V + p6*X^3 + p7*V^3 + p8*cos(t) + p9*sin(t)
    
    Inputs are scaled to O(1) to improve conditioning.
    """
    # Scale inputs
    # t range [30, 50]
    # x range [-0.2056, 0.1749] -> scale by 0.2
    # v range [-0.3391, 0.3624] -> scale by 0.35
    
    X = x / 0.2
    V = v / 0.35
    
    # Compute basis terms
    term_0 = np.ones_like(t)
    term_1 = X
    term_2 = V
    term_3 = X * X
    term_4 = V * V
    term_5 = X * V
    term_6 = X * X * X
    term_7 = V * V * V
    term_8 = np.cos(t)
    term_9 = np.sin(t)
    
    # Combine with parameters
    pred = (params[0] * term_0 + 
            params[1] * term_1 + 
            params[2] * term_2 + 
            params[3] * term_3 + 
            params[4] * term_4 + 
            params[5] * term_5 + 
            params[6] * term_6 + 
            params[7] * term_7 + 
            params[8] * term_8 + 
            params[9] * term_9)
            
    return pred

# EVOLVE-BLOCK-END
