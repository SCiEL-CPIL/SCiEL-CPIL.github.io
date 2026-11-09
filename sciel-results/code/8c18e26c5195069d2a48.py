import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    # Refined Cohn-Goncalves Laguerre construction for Fourier sign uncertainty.
    # m = 12 double roots, n = 1 dimension.
    # The double-root locations are the zeros of the generalized Laguerre
    # polynomial L_m^{(2n-1)}(x) = L_{12}^{(1)}(x).
    m = 12
    n = 1
    alpha = 2 * n - 1  # alpha = 1
    roots = roots_genlaguerre(m, float(alpha))[0]
    roots_sorted = sorted(float(r) for r in roots)
    return roots_sorted
