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
    
    # 10-term hybrid basis: 2 cos + 2 sin (freqs 0.5, 1.0) + 6 polynomial (1,x,x^2,x^3,x^4,x^5)
    # Lower frequencies (0.5, 1.0) better capture broad periodic structure over [-3,3]
    # Degree-5 polynomial captures wide variety of smooth functions
    freqs = np.array([0.5, 1.0])
    
    # Compute basis functions
    cos_basis = np.cos(np.outer(x, freqs))  # (n, 2)
    sin_basis = np.sin(np.outer(x, freqs))  # (n, 2)
    
    # Polynomial basis: 1, x, x^2, x^3, x^4, x^5
    poly_basis = np.column_stack([
        np.ones_like(x),
        x,
        x**2,
        x**3,
        x**4,
        x**5
    ])  # (n, 6)
    
    # params[0:2] -> cos, params[2:4] -> sin, params[4:10] -> poly
    cos_coeffs = params[0:2]
    sin_coeffs = params[2:4]
    poly_coeffs = params[4:10]
    
    # prediction
    pred = cos_basis @ cos_coeffs + sin_basis @ sin_coeffs + poly_basis @ poly_coeffs
    
    return pred
# EVOLVE-BLOCK-END
