"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series model with 5 sine and 5 cosine terms (frequencies 1-5)
    plus a constant term. 11 parameters total, but we only have 10.
    
    We'll use:
    - 1 constant term
    - 4 cosine terms (freq 1-4)
    - 5 sine terms (freq 1-5)
    Total: 10 parameters
    
    Actually, let's use frequencies 1-5 for both, but we only have 10 params.
    Let's do:
    - params[0]: constant
    - params[1:6]: cosine coefficients for freq 1-5
    - params[6:10]: sine coefficients for freq 1-4
    That's 10 params.
    
    Wait, let me reconsider. The task says params is an array of length 10.
    Let's use:
    - params[0]: constant
    - params[1:6]: cosine coefficients for freq 1-5 (5 terms)
    - params[6:10]: sine coefficients for freq 1-4 (4 terms)
    Total: 1 + 5 + 4 = 10
    
    This gives a reasonable Fourier basis on [-3, 3].
    """
    # Build Fourier basis
    # Frequencies: 1, 2, 3, 4, 5 for cosine; 1, 2, 3, 4 for sine
    # x is in [-3, 3]
    
    # Precompute sin and cos for each frequency
    # cos(k*x) for k = 1..5
    # sin(k*x) for k = 1..4
    
    result = params[0] * np.ones_like(x)
    
    # Cosine terms: freq 1-5
    for k in range(1, 6):
        result = result + params[k] * np.cos(k * x)
    
    # Sine terms: freq 1-4
    for k in range(1, 5):
        result = result + params[5 + k] * np.sin(k * x)
    
    return result
# EVOLVE-BLOCK-END
