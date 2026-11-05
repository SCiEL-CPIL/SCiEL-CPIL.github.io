import numpy as np
from scipy.special import roots_laguerre

def construct():
    m = 12
    # Use alpha=0 for standard Laguerre polynomial in 1D
    # Roots x_i of L_m^(0)(x)
    x_roots, _ = roots_laguerre(m, 0)
    # The double root locations r_i are sqrt(x_i)
    r_roots = np.sqrt(x_roots)
    return list(r_roots)
