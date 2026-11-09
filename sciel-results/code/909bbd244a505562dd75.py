import numpy as np
from scipy.special import genlaguerre
import time

def construct():
    """
    Return 12 double-root locations z_1..z_12.
    
    Strategy: Use the 12 smallest positive zeros of the Bessel function J_{1/2}(x),
    which are the roots of sin(sqrt(x))/sqrt(x). These are at x = (k*pi)^2 for k=1..12.
    This exploits the asymptotic orthogonality of Laguerre polynomials to the Bessel measure.
    
    We verify that this configuration is valid (positive, finite, distinct) and return it.
    """
    # The zeros of J_{1/2}(x) are at x = (k*pi)^2 for k = 1, 2, 3, ...
    # J_{1/2}(x) = sqrt(2/(pi*x)) * sin(sqrt(x))
    # So the zeros (excluding x=0) are at sqrt(x) = k*pi, i.e., x = (k*pi)^2
    
    k_vals = np.arange(1, 13, dtype=np.float64)
    z_vals = (k_vals * np.pi) ** 2
    
    # Check validity: all positive, finite, distinct
    if not np.all(np.isfinite(z_vals)):
        raise ValueError("Non-finite values in z")
    if not np.all(z_vals > 0):
        raise ValueError("Non-positive values in z")
    if len(set(z_vals)) != 12:
        raise ValueError("Values in z are not distinct")
    
    return [float(z) for z in z_vals]
