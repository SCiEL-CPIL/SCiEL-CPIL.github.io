"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(x)

    x = np.asarray(x, dtype=np.float64)
    
    # 10-term basis: 5 cosines + 5 sines with frequencies {1,2,3,4,5}
    # This is linear in params, so BFGS will converge quickly
    freqs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Compute basis functions
    # shape: (n_points, 5) for cos, (n_points, 5) for sin
    cos_basis = np.cos(np.outer(x, freqs))
    sin_basis = np.sin(np.outer(x, freqs))
    
    # params[0:5] multiply cos terms, params[5:10] multiply sin terms
    cos_coeffs = params[0:5]
    sin_coeffs = params[5:10]
    
    # prediction = sum_i cos_coeffs[i] * cos(freqs[i]*x) + sum_i sin_coeffs[i] * sin(freqs[i]*x)
    pred = cos_basis @ cos_coeffs + sin_basis @ sin_coeffs
    
    return pred
# EVOLVE-BLOCK-END
