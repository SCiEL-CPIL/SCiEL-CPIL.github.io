"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Multi-scale Fourier basis approximation.
    
    Strategy:
    Use 10 parameters to weight a combination of low-frequency and high-frequency
    sine/cosine terms. This allows the model to capture periodic structures 
    and oscillations that a polynomial or simple RBF might miss or approximate poorly.
    
    Basis functions:
    1. cos(pi * x)
    2. sin(pi * x)
    3. cos(pi * y)
    4. sin(pi * y)
    5. cos(2*pi * x)
    6. sin(2*pi * x)
    7. cos(2*pi * y)
    8. sin(2*pi * y)
    9. cos(pi * x * y)
    10. 1.0 (bias)
    
    This set covers fundamental frequencies and some harmonics, plus an interaction term.
    The use of pi scaling ensures the basis functions have full periods across the [-1, 1] domain.
    """
    p = params
    
    # Precompute trigonometric terms
    # Using numpy vectorized operations for efficiency
    
    # Low frequency (fundamental)
    c1 = np.cos(np.pi * x)
    s1 = np.sin(np.pi * x)
    c2 = np.cos(np.pi * y)
    s2 = np.sin(np.pi * y)
    
    # High frequency (2nd harmonic)
    c3 = np.cos(2.0 * np.pi * x)
    s3 = np.sin(2.0 * np.pi * x)
    c4 = np.cos(2.0 * np.pi * y)
    s4 = np.sin(2.0 * np.pi * y)
    
    # Interaction term
    c5 = np.cos(np.pi * x * y)
    
    # Bias
    b = 1.0
    
    # Combine
    pred = (
        p[0] * c1 + 
        p[1] * s1 + 
        p[2] * c2 + 
        p[3] * s2 + 
        p[4] * c3 + 
        p[5] * s3 + 
        p[6] * c4 + 
        p[7] * s4 + 
        p[8] * c5 + 
        p[9] * b
    )
    
    return pred
# EVOLVE-BLOCK-END
