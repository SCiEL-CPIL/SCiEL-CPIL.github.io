"""Symbolic regression with 1 input(s) using Chebyshev polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the unknown function using a linear combination of Chebyshev polynomials.
    The input x is mapped to t in [-1, 1] to utilize the stable Chebyshev basis.
    """
    x = np.asarray(x, dtype=np.float64)
    
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
        
    # Map x from [0.1, 6] to t in [-1, 1]
    # t = (2*x - (min+max)) / (max-min)
    # min=0.1, max=6.0
    t = (2.0 * x - 6.1) / 5.9
    
    # Clip t to [-1, 1] to prevent numerical drift outside the domain
    # which can cause Chebyshev polynomials to grow exponentially and cause instability
    t = np.clip(t, -1.0, 1.0)
    
    # Hybrid Basis: 5 Chebyshev + 5 Fourier
    n_points = x.shape[0]
    if n_points == 0:
        return np.array([])
        
    basis = np.zeros((n_points, 10), dtype=np.float64)
    
    # First 5 columns: Chebyshev T_0 to T_4
    basis[:, 0] = 1.0
    basis[:, 1] = t
    
    for k in range(2, 5):
        basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
        
    # Next 5 columns: Fourier terms
    # Using frequencies 1, 2, 3 on t in [-1, 1]
    # sin(pi*t), cos(pi*t), sin(2*pi*t), cos(2*pi*t), sin(3*pi*t)
    basis[:, 5] = np.sin(np.pi * t)
    basis[:, 6] = np.cos(np.pi * t)
    basis[:, 7] = np.sin(2.0 * np.pi * t)
    basis[:, 8] = np.cos(2.0 * np.pi * t)
    basis[:, 9] = np.sin(3.0 * np.pi * t)
            
    # Linear combination
    # params is shape (10,)
    pred = basis @ params
    
    return pred
# EVOLVE-BLOCK-END
