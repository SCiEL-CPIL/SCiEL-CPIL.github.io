"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a structured physical basis.
    
    The basis is designed around known oscillator physics:
    - Linear damping: v
    - Linear restoring: x
    - Cubic restoring (Duffing): x^3
    - Quadratic damping (Van der Pol): x*v
    - Higher-order corrections
    
    We use a mixed basis with physics-inspired terms and some flexible terms.
    
    params: 10 fitted constants
    """
    p = params
    
    # Basic terms
    t1 = x                          # linear restoring
    t2 = v                          # linear damping
    t3 = x * x * x                  # Duffing cubic
    t4 = x * v                      # Van der Pol quadratic damping
    t5 = x * x                      # quadratic restoring
    t6 = v * v                      # quadratic velocity
    t7 = x * x * v                  # cubic mixed
    t8 = x * v * v                  # cubic mixed
    t9 = x * x * x * v              # quartic mixed
    t10 = x * x * x * x             # quartic restoring
    
    return (p[0] * t1 + 
            p[1] * t2 + 
            p[2] * t3 + 
            p[3] * t4 + 
            p[4] * t5 + 
            p[5] * t6 + 
            p[6] * t7 + 
            p[7] * t8 + 
            p[8] * t9 + 
            p[9] * t10)
# EVOLVE-BLOCK-END
