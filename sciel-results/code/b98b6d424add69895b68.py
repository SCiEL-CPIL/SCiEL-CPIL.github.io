"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: The evaluator kills after 150s.
# We set a conservative limit to ensure we return within time.
_MAX_TIME = 140.0
_START_TIME = time.time()

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of a diverse, smooth mixed basis of functions.
    
    The evaluator fits the 10 params via BFGS. We provide a diverse set of
    basis functions that can approximate many common hidden functions
    (polynomials, trigonometric, hyperbolic, Gaussian bumps, logarithmic, etc.).
    
    Basis functions:
    0: 1 (constant)
    1: x
    2: x^2
    3: x^3
    4: x^4
    5: sin(x)
    6: cos(x)
    7: exp(-0.5 * x^2)  (Gaussian bump)
    8: tanh(x)          (Sigmoid-like saturation)
    9: log(1 + x^2)     (Slow-growing, well-behaved, captures curvature)
    """
    # Check time limit
    if time.time() - _START_TIME > _MAX_TIME:
        # Return a safe default if we are about to time out
        return np.zeros_like(x)

    x = np.asarray(x, dtype=np.float64)
    
    # Compute basis functions
    # Vectorized operations for speed
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = x * x
    b3 = b2 * x
    b4 = b2 * b2
    
    b5 = np.sin(x)
    b6 = np.cos(x)
    
    # exp(-0.5 * x^2) is stable and bounded
    b7 = np.exp(-0.5 * b2)
    
    # tanh(x) is stable and bounded
    b8 = np.tanh(x)
    
    # log(1 + x^2) is well-behaved on [-3, 3]
    b9 = np.log1p(b2)
    
    # Linear combination
    # params is length 10
    # p0*b0 + p1*b1 + ... + p9*b9
    
    pred = (params[0] * b0 + 
            params[1] * b1 + 
            params[2] * b2 + 
            params[3] * b3 + 
            params[4] * b4 + 
            params[5] * b5 + 
            params[6] * b6 + 
            params[7] * b7 + 
            params[8] * b8 + 
            params[9] * b9)
            
    return pred
# EVOLVE-BLOCK-END
