"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale input to [-1, 1] for better conditioning
    # x is in [-3, 3], so u = x / 3 is in [-1, 1]
    u = x / 3.0
    
    # Mixed basis: polynomials + trigonometric + exponential
    # All terms scaled to O(1) magnitude for good Hessian conditioning
    
    # Polynomial terms (Legendre-like, scaled)
    term_0 = np.ones_like(u)                    # P_0
    term_1 = u                                  # P_1
    term_2 = (3.0 * u**2 - 1.0) / 2.0           # P_2
    
    # Trigonometric terms (scaled to O(1))
    term_3 = np.sin(u)
    term_4 = np.cos(u) - 1.0                    # Centered cosine
    term_5 = np.sin(2.0 * u)
    term_6 = np.cos(2.0 * u) - 1.0              # Centered cosine
    
    # Exponential terms (scaled to O(1) on [-1,1])
    term_7 = np.exp(u) - 1.0                    # Centered exp
    term_8 = np.exp(-u) - 1.0                   # Centered exp
    term_9 = u * np.exp(u**2 / 2.0)             # Gaussian-like weighted
    
    # Combine with params
    c = params
    
    return (c[0] * term_0 + 
            c[1] * term_1 + 
            c[2] * term_2 + 
            c[3] * term_3 + 
            c[4] * term_4 + 
            c[5] * term_5 + 
            c[6] * term_6 + 
            c[7] * term_7 + 
            c[8] * term_8 + 
            c[9] * term_9)

# EVOLVE-BLOCK-END
