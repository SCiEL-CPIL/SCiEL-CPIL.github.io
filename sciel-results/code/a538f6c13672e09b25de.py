"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale input to [-1, 1] to improve numerical stability for higher powers
    t = x / 3.0
    
    # Construct basis functions: 1, t, t^2, ..., t^9
    # We can compute powers efficiently
    # Or just use a loop/vectorized approach
    # Since x is an array, we can compute t**i for i in 0..9
    
    # To be efficient and avoid repeated power calculations if possible, 
    # but for N=200 and degree 9, it's negligible.
    
    # params[0] * 1
    # params[1] * t
    # ...
    # params[9] * t**9
    
    # Vectorized construction
    # We can use np.power.outer or a loop
    # Let's use a simple loop or broadcasting
    
    # t is shape (N,)
    # We want result shape (N,)
    
    # p[0] + p[1]*t + p[2]*t^2 ...
    
    # Using Horner's method for stability and speed?
    # f(t) = p[0] + t*(p[1] + t*(p[2] + ... + t*p[9]))
    
    # Horner's method is numerically more stable for polynomials.
    result = np.zeros_like(x)
    for i in range(9, -1, -1):
        result = result * t + params[i]
        
    return result
# EVOLVE-BLOCK-END
