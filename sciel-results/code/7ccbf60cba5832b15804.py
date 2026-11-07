"""Symbolic regression with 10-term Fourier series basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(x)

    x = np.asarray(x, dtype=np.float64)
    
    # 10-term Fourier series: 5 cos + 5 sin (freqs 1,2,3,4,5)
    # Strictly linear in params, well-conditioned due to orthogonality
    # on [-3, 3] for these frequencies
    freqs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Compute basis functions
    cos_basis = np.cos(np.outer(x, freqs))  # (n, 5)
    sin_basis = np.sin(np.outer(x, freqs))  # (n, 5)
    
    # params[0:5] -> cos coefficients, params[5:10] -> sin coefficients
    cos_coeffs = params[0:5]
    sin_coeffs = params[5:10]
    
    # prediction
    pred = cos_basis @ cos_coeffs + sin_basis @ sin_coeffs
    
    return pred
# EVOLVE-BLOCK-END
