"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Use a 10-term Fourier series with fixed frequencies {1,2,3,4,5} as a strictly
    linear-in-params basis. The trigonometric functions are well-conditioned on
    [-3, 3], allowing BFGS to converge reliably.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Fourier basis: sin and cos for frequencies 1 through 5
    # This gives 10 basis functions total
    freqs = np.array([1, 2, 3, 4, 5])
    
    sin_terms = np.column_stack([np.sin(f * x) for f in freqs])
    cos_terms = np.column_stack([np.cos(f * x) for f in freqs])
    
    basis = np.column_stack([sin_terms, cos_terms])
    
    return basis @ params
# EVOLVE-BLOCK-END
