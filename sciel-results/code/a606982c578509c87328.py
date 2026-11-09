import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    # Compute the 12 roots of L_12^(0)(x)
    roots = roots_genlaguerre(12, 0)
    # Ensure they are positive and sorted
    roots = np.sort(roots)
    return [float(r) for r in roots]
