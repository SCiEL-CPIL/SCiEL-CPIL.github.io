import numpy as np
from scipy.special import jn_zeros

def construct():
    """
    Return 12 double-root locations z_1..z_12.
    
    Strategy: Use the 12 smallest positive zeros of the Bessel function J_{1/2}(x).
    Calculated via scipy.special.jn_zeros to ensure standard numerical derivation.
    """
    z_vals = jn_zeros(0.5, 12)
    z_list = [float(z) for z in z_vals]
    return z_list
