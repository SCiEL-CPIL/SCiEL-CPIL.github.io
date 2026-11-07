"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model using a mixed basis of polynomial and periodic terms.
    The basis is designed to capture:
    - Conservative forces (odd powers of x)
    - Damping forces (odd powers of v)
    - Cross-coupling terms
    - Time-periodic forcing (cos/sin of t and 2t)
    
    Basis functions (10 params):
    0: 1
    1: x
    2: x^3
    3: x^5
    4: v
    5: v^3
    6: x*v
    7: cos(t)
    8: sin(t)
    9: cos(2t)
    """
    p = params
    
    # Ensure inputs are arrays for vectorized operations
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    
    # Polynomial terms in x (conservative force)
    term0 = p[0]
    term1 = p[1] * x
    term2 = p[2] * x**3
    term3 = p[3] * x**5
    
    # Polynomial terms in v (damping)
    term4 = p[4] * v
    term5 = p[5] * v**3
    
    # Cross term
    term6 = p[6] * x * v
    
    # Periodic forcing terms
    term7 = p[7] * np.cos(t)
    term8 = p[8] * np.sin(t)
    term9 = p[9] * np.cos(2 * t)
    
    return term0 + term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
# EVOLVE-BLOCK-END
