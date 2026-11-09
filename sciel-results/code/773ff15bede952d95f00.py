import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    # Use the roots of the generalized Laguerre polynomial L_12^(-1/2)(x)
    # as the double-root locations. These are the orthogonal polynomial nodes
    # for the weight x^(-1/2) e^(-x), which may provide a more optimal
    # distribution for minimizing the largest root of the quotient q(x).
    nodes, weights = roots_genlaguerre(12, -0.5)
    # Ensure they are returned as a list of floats, sorted
    return sorted([float(z) for z in nodes])
