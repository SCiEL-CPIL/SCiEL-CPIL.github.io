import time
import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    start_time = time.time()
    deadline = 600.0 - 30.0  # safety margin
    
    # Compute the 12 zeros of the generalized Laguerre polynomial L_12^(1)(x)
    # m=12, n=1 => alpha=1
    n_roots = 12
    alpha = 1
    
    # roots_genlaguerre returns zeros and weights for L_n^(alpha)(x)
    # The zeros are the optimal root locations for the Cohn-Goncales construction
    zeros, weights = roots_genlaguerre(n_roots, alpha)
    
    # Return the 12 positive double-root locations
    return [float(z) for z in zeros]
