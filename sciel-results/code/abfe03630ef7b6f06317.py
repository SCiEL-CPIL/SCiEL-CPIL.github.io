import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    # The task specifies m=12. The optimal nodes for the refined Cohn-Goncalves
    # construction are often associated with the generalized Laguerre polynomials.
    # Using alpha=1 corresponds to the roots of the derivative of L_13 or L_12^(1).
    # This is a common improvement over the standard alpha=0 roots for uncertainty bounds
    # involving specific weight functions or sign change constraints.
    roots, _ = roots_genlaguerre(12, 1)
    # Return as a list of floats
    return [float(r) for r in roots]
