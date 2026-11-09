import numpy as np
import time

def construct():
    """
    Return 12 double-root locations z_1..z_12.
    
    Strategy: Use the 12 smallest positive zeros of the Bessel function J_{1/2}(x),
    which are the exact roots of sin(sqrt(x))/sqrt(x), to define z_i.
    These are at x = (k*pi)^2 for k=1..12.
    
    This exploits the asymptotic orthogonality of Laguerre polynomials to the Bessel measure,
    potentially confining the quotient q(x)'s largest root B more effectively than finite-dimensional orthogonal nodes.
    """
    k_vals = np.arange(1, 13, dtype=np.float64)
    z_vals = (k_vals * np.pi) ** 2
    return [float(z) for z in z_vals]
