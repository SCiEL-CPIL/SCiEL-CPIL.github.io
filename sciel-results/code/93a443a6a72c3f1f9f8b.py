"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation using a Fourier series expansion on [-1, 1]^2.
    Uses sine and cosine terms of various frequencies to capture
    periodic/oscillatory behavior that polynomials approximate poorly.
    
    params: array of length 10, fitted by BFGS.
    """
    # Build a Fourier basis with 10 parameters
    # Frequencies: 1, 2, 3 (scaled to [-1,1] domain)
    # Terms:
    #   p0: constant
    #   p1: sin(pi*x)
    #   p2: cos(pi*x)
    #   p3: sin(pi*y)
    #   p4: cos(pi*y)
    #   p5: sin(pi*(x+y))
    #   p6: cos(pi*(x+y))
    #   p7: sin(pi*(x-y))
    #   p8: cos(pi*(x-y))
    #   p9: sin(2*pi*x) * cos(2*pi*y)  (interaction term)
    
    pi = np.pi
    
    t0 = params[0]
    t1 = params[1] * np.sin(pi * x)
    t2 = params[2] * np.cos(pi * x)
    t3 = params[3] * np.sin(pi * y)
    t4 = params[4] * np.cos(pi * y)
    t5 = params[5] * np.sin(pi * (x + y))
    t6 = params[6] * np.cos(pi * (x + y))
    t7 = params[7] * np.sin(pi * (x - y))
    t8 = params[8] * np.cos(pi * (x - y))
    t9 = params[9] * np.sin(2 * pi * x) * np.cos(2 * pi * y)
    
    return t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9
# EVOLVE-BLOCK-END
