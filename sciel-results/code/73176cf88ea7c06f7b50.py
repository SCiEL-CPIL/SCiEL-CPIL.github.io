# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-informed model for nonlinear oscillator acceleration.
    
    Structure: a = c0 + c1*x + c2*x^3 + c3*v + c4*v^3 + c5*v^5 + c6*v^7 + c7*x^2*v + c8*x*v^2 + c9*sin(x)
    
    Parameters:
    c0: bias
    c1: linear spring
    c2: cubic spring (Duffing)
    c3: linear damping
    c4: cubic damping
    c5: quintic damping
    c6: septic damping (for stability at high v)
    c7: quadratic spring-velocity coupling
    c8: linear spring-velocity coupling
    c9: sine term (for periodicity/symmetry)
    """
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    v5 = v3 * v2
    v7 = v5 * v2
    
    term1 = params[0]
    term2 = params[1] * x
    term3 = params[2] * x3
    term4 = params[3] * v
    term5 = params[4] * v3
    term6 = params[5] * v5
    term7 = params[6] * v7
    term8 = params[7] * x2 * v
    term9 = params[8] * x * v2
    term10 = params[9] * np.sin(x)
    
    return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10

# EVOLVE-BLOCK-END
