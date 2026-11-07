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
    
    # 10-term hybrid basis: 3 cos + 3 sin (freqs 1,2,3) + 4 polynomial (1,x,x^2,x^3)
    # This captures both periodic and polynomial components
    freqs = np.array([1.0, 2.0, 3.0])
    
    # Compute basis functions
    cos_basis = np.cos(np.outer(x, freqs))  # (n, 3)
    sin_basis = np.sin(np.outer(x, freqs))  # (n, 3)
    
    # Polynomial basis: 1, x, x^2, x^3
    poly_basis = np.column_stack([
        np.ones_like(x),
        x,
        x**2,
        x**3
    ])  # (n, 4)
    
    # params[0:3] -> cos, params[3:6] -> sin, params[6:10] -> poly
    cos_coeffs = params[0:3]
    sin_coeffs = params[3:6]
    poly_coeffs = params[6:10]
    
    # prediction
    pred = cos_basis @ cos_coeffs + sin_basis @ sin_coeffs + poly_basis @ poly_coeffs
    
    return pred
# EVOLVE-BLOCK-END
