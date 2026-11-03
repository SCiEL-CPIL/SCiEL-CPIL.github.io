import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    n = 12
    alpha = 1
    # roots_genlaguerre returns (x, w) where x are the roots
    roots, _ = roots_genlaguerre(n, alpha)
    # Ensure they are sorted and positive
    roots = sorted(roots)
    return [float(r) for r in roots]
