"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_start_time = time.time()
DEADLINE = 150.0  # Wall-clock limit
SAFETY_MARGIN = 5.0  # Seconds to spare

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Separable multiplicative basis using degree-1 polynomials in each variable.
    This basis exactly represents functions of the form f(x)f(y)f(z) and captures
    multiplicative coupling that additive polynomial bases approximate poorly.
    
    Basis (10 terms):
      p[0]:  1
      p[1]:  u
      p[2]:  v
      p[3]:  w
      p[4]:  u*v
      p[5]:  u*w
      p[6]:  v*w
      p[7]:  u*v*w
      p[8]:  u*u
      p[9]:  v*v
    
    Note: We include u*u and v*v to provide some quadratic flexibility in the
    univariate directions, while maintaining the core multiplicative structure.
    The inputs are normalized to [-1, 1] for numerical stability.
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > DEADLINE - SAFETY_MARGIN:
        # Fallback: return a constant (p[0]) to be safe
        return np.ones_like(x) * params[0]
    
    p = params
    
    # Normalize inputs to [-1, 1]
    # Inputs are in [0.5, 2.0]
    # center = 1.25, half_range = 0.75
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    # Basis functions
    # 1. Constant
    b0 = np.ones_like(u)
    
    # 2-4. Linear terms
    b1 = u
    b2 = v
    b3 = w
    
    # 5-7. Pairwise multiplicative interactions
    b4 = u * v
    b5 = u * w
    b6 = v * w
    
    # 8. Triple multiplicative interaction
    b7 = u * v * w
    
    # 9-10. Squared terms for univariate flexibility
    b8 = u * u
    b9 = v * v
    
    # Combine
    return (p[0] * b0 + 
            p[1] * b1 + 
            p[2] * b2 + 
            p[3] * b3 + 
            p[4] * b4 + 
            p[5] * b5 + 
            p[6] * b6 + 
            p[7] * b7 + 
            p[8] * b8 + 
            p[9] * b9)
# EVOLVE-BLOCK-END
