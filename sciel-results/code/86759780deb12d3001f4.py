"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series with 5 sine and 5 cosine terms.
    
    Parameters (10 total):
      p[0:5]  - amplitudes of cos(k*x) for k=1..5
      p[5:10] - amplitudes of sin(k*x) for k=1..5
    
    This gives a periodic basis that can capture oscillatory structures
    efficiently. Since the input range is [-2, 2], the fundamental frequency
    1 rad/unit gives a period of ~6.28, which covers the range well.
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Frequencies: k = 1, 2, 3, 4, 5
    k = np.arange(1, 6, dtype=np.float64)
    
    # Compute cos and sin terms
    # k is shape (5,), x is shape (N,)
    # We need outer product: k[:, None] * x[None, :]
    kx = np.outer(k, x)
    
    cos_terms = np.cos(kx)  # shape (5, N)
    sin_terms = np.sin(kx)  # shape (5, N)
    
    # Weighted sum
    cos_amp = p[0:5]  # shape (5,)
    sin_amp = p[5:10]  # shape (5,)
    
    y = np.sum(cos_amp[:, None] * cos_terms, axis=0) + np.sum(sin_amp[:, None] * sin_terms, axis=0)
    
    return y
# EVOLVE-BLOCK-END
