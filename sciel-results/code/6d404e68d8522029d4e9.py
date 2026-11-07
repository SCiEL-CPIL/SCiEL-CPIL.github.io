import numpy as np
import time

# EVOLVE-BLOCK-START

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model using a cubic polynomial basis in position and velocity.
    
    Model:
    a = p0 + p1*X + p2*V + p3*X^2 + p4*V^2 + p5*X*V 
      + p6*X^3 + p7*V^3 + p8*X^2*V + p9*X*V^2
      
    Inputs are scaled to O(1) range to improve conditioning.
    X = x / 0.2
    V = v / 0.3
    
    This structure is linear in parameters, allowing BFGS to converge quickly
    to the optimal linear combination of these basis functions.
    """
    # Scale inputs to O(1) range based on training data bounds
    X = x / 0.2
    V = v / 0.3
    
    # Extract parameters
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Compute basis terms
    term0 = np.ones_like(X)
    term1 = X
    term2 = V
    term3 = X**2
    term4 = V**2
    term5 = X * V
    term6 = X**3
    term7 = V**3
    term8 = X**2 * V
    term9 = X * V**2
    
    # Combine
    pred = (p0 * term0 + p1 * term1 + p2 * term2 + p3 * term3 + p4 * term4 +
            p5 * term5 + p6 * term6 + p7 * term7 + p8 * term8 + p9 * term9)
            
    return pred

# EVOLVE-BLOCK-END
