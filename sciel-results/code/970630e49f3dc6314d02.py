"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure x is a 1D array for safe operations
    x = np.asarray(x, dtype=np.float64)
    
    # Scale x to [-1, 1] for better numerical conditioning
    t = x / 3.0
    
    # Use Legendre polynomials (orthogonal on [-1,1]) for well-conditioned basis
    # P0=1, P1=t, P2=(3t^2-1)/2, P3=(5t^3-3t)/2, P4=(35t^4-30t^2+3)/8
    # P5=(63t^5-70t^3+15t)/8, P6=(231t^6-315t^4+105t^2-5)/16
    # P7=(429t^7-693t^5+315t^3-35t)/16, P8=(6435t^8-12012t^6+6930t^4-1260t^2+35)/128
    # P9=(12155t^9-25740t^7+18018t^5-4620t^3+315t)/128
    
    f0 = np.ones_like(t)
    f1 = t
    f2 = (3*t**2 - 1) / 2
    f3 = (5*t**3 - 3*t) / 2
    f4 = (35*t**4 - 30*t**2 + 3) / 8
    f5 = (63*t**5 - 70*t**3 + 15*t) / 8
    f6 = (231*t**6 - 315*t**4 + 105*t**2 - 5) / 16
    f7 = (429*t**7 - 693*t**5 + 315*t**3 - 35*t) / 16
    f8 = (6435*t**8 - 12012*t**6 + 6930*t**4 - 1260*t**2 + 35) / 128
    f9 = (12155*t**9 - 25740*t**7 + 18018*t**5 - 4620*t**3 + 315*t) / 128
    
    # Stack features into a matrix of shape (N, 10)
    features = np.column_stack((f0, f1, f2, f3, f4, f5, f6, f7, f8, f9))
    
    # Linear combination
    return features.dot(params)

# EVOLVE-BLOCK-END
