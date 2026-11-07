"""Symbolic regression with 3 inputs – single-hidden-layer tanh network basis."""
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0 # 10s safety margin before 120s limit

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the function using a single-hidden-layer tanh network with 10 parameters.
    
    Architecture:
      - 2 Hidden Units
      - Hidden Unit 1: tanh(p0*x + p1*y + p2*z + p6)
      - Hidden Unit 2: tanh(p3*x + p4*y + p5*z + p7)
      - Output: p8 * h1 + p9 * h2
      
    Parameters:
      p0, p1, p2: Weights for x, y, z in hidden unit 1
      p3, p4, p5: Weights for x, y, z in hidden unit 2
      p6: Bias for hidden unit 1
      p7: Bias for hidden unit 2
      p8: Output weight for hidden unit 1
      p9: Output weight for hidden unit 2
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return zeros or a simple fallback to avoid crash, though this shouldn't happen
        return np.zeros_like(x)

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute hidden layer activations
    h1 = np.tanh(p0 * x + p1 * y + p2 * z + p6)
    h2 = np.tanh(p3 * x + p4 * y + p5 * z + p7)
    
    # Output: linear combination of hidden units
    return p8 * h1 + p9 * h2
