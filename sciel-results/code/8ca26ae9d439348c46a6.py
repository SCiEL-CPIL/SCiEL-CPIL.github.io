import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    # The task specifies m=12 and dimension n=1.
    # In the context of Cohn-Goncalves for 1D Fourier uncertainty,
    # the associated Laguerre parameter is often alpha=0 (standard Laguerre).
    # The previous attempt used alpha=1, which corresponds to a different weight.
    roots, weights = roots_genlaguerre(12, 0)
    
    # Return the roots as a list of floats
    return [float(r) for r in roots]
