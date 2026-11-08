import numpy as np
from scipy.special import roots_laguerre

def construct():
    # m=12
    m = 12
    # Get roots of L_m(x)
    roots, _ = roots_laguerre(m)
    # Ensure they are positive and sorted
    roots = sorted(roots)
    return roots
