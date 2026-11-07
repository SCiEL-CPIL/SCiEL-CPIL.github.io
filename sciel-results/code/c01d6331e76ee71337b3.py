"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers a function using a Fourier Series Basis.
    
    The model is linear in the 10 parameters:
    f(x, y) = p0 + p1*sin(pi*x) + p2*cos(pi*x) + p3*sin(pi*y) + p4*cos(pi*y) +
              p5*sin(2*pi*x) + p6*cos(2*pi*x) + p7*sin(2*pi*y) + p8*cos(2*pi*y) +
              p9*sin(pi*(x+y))
    
    This basis is chosen to capture smooth, periodic, and interaction structures
    which are common in hidden target functions on bounded domains [-1, 1].
    It provides a better approximation space for non-polynomial functions than
    low-degree Chebyshev or monomial bases.
    """
    # Convert inputs to float64 for precision
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Precompute basis functions
    # 1. Constant
    b0 = np.ones_like(x)
    
    # 2. First harmonic x
    b1 = np.sin(np.pi * x)
    b2 = np.cos(np.pi * x)
    
    # 3. First harmonic y
    b3 = np.sin(np.pi * y)
    b4 = np.cos(np.pi * y)
    
    # 4. Second harmonic x
    b5 = np.sin(2.0 * np.pi * x)
    b6 = np.cos(2.0 * np.pi * x)
    
    # 5. Second harmonic y
    b7 = np.sin(2.0 * np.pi * y)
    b8 = np.cos(2.0 * np.pi * y)
    
    # 6. Interaction term (sum frequency)
    b9 = np.sin(np.pi * (x + y))
    
    # Compute predictions: linear combination
    pred = (
        p[0] * b0 +
        p[1] * b1 +
        p[2] * b2 +
        p[3] * b3 +
        p[4] * b4 +
        p[5] * b5 +
        p[6] * b6 +
        p[7] * b7 +
        p[8] * b8 +
        p[9] * b9
    )
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(pred)):
        # Fallback to a simple linear model if numerical issues arise
        # This is unlikely with bounded inputs and trig functions, but safe
        pred = p[0] + p[1] * x + p[2] * y + p[3] * (x * y)
        if not np.all(np.isfinite(pred)):
            return np.zeros_like(x)
            
    return pred
# EVOLVE-BLOCK-END
