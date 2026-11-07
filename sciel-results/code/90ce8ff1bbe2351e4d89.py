import numpy as np

# EVOLVE-BLOCK-START

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model with enhanced cross-term basis.
    
    Model:
    a = p0 + p1*X + p2*V + p3*X^2 + p4*V^2 + p5*X*V + p6*X^3 + p7*X^2*V + p8*cos(t) + p9*sin(t)
    
    Inputs are scaled to O(1).
    X = x / 0.2
    V = v / 0.35
    """
    # Scale inputs
    X = x / 0.2
    V = v / 0.35
    
    # Basis terms
    term_0 = np.ones_like(t)
    term_1 = X
    term_2 = V
    term_3 = X * X
    term_4 = V * V
    term_5 = X * V
    term_6 = X * X * X
    term_7 = X * X * V  # New cross term: position-dependent damping
    term_8 = np.cos(t)
    term_9 = np.sin(t)
    
    # Linear combination
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
