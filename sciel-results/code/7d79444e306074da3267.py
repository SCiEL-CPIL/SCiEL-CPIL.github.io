import numpy as np
from scipy.special import roots_laguerre

def construct():
        # Get roots of L_12^(1)(x)
        # roots_laguerre(n) gives roots of L_n^(0).
        # We need L_n^(alpha).
        # We can use numpy.polynomial.laguerre.Laguerre.basis or similar?
        # Or scipy.special.eval_laguerre? No, we need roots.
        # There is no direct scipy function for roots_laguerre(n, alpha).
        # But we can construct the polynomial and find roots.
        
        # Alternatively, the seed was arithmetic. 
        # The knowledge says "arithmetic... was part of the grid search".
        # And "optimal root scale around 10x the seed".
        # This phrasing "scale... the seed" implies keeping the *shape* of the seed (arithmetic) and just scaling it.
        
        return [40.0 * (i + 1) for i in range(12)]
