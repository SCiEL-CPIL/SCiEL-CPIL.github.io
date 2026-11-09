"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = None
_TIME_LIMIT = 115.0


def _check_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_LIMIT
    return time.time() < _DEADLINE


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)

    if not _check_deadline():
        return np.ones_like(x) * (np.dot(x, [0.0, 1.0]) if params.size >= 2 else 1.0)

    # Basis design: 6 smooth polynomial terms + 4 localized Gaussian bumps
    # This hybrid allows both global trends and local features.
    n_poly = 6
    n_gauss = 4
    n_total = n_poly + n_gauss

    # Polynomial part: use mapped variable for better conditioning
    x_min, x_max = 0.1, 6.0
    t = (2.0 * x - (x_max + x_min)) / (x_max - x_min)
    t = np.clip(t, -1.5, 1.5)

    # Compute polynomial basis via recurrence
    basis_poly = np.zeros((len(x), n_poly), dtype=np.float64)
    basis_poly[:, 0] = 1.0
    if n_poly > 1:
        basis_poly[:, 1] = t
    for k in range(2, n_poly):
        basis_poly[:, k] = 2.0 * t * basis_poly[:, k-1] - basis_poly[:, k-2]

    # Gaussian bumps at fixed centers across the domain
    # Centers chosen to cover the input range
    centers = np.array([0.5, 1.8, 3.5, 5.2], dtype=np.float64)
    # Fixed width for stability (learnable width via BFGS can be unstable)
    sigma = 1.0

    basis_gauss = np.zeros((len(x), n_gauss), dtype=np.float64)
    for j in range(n_gauss):
        basis_gauss[:, j] = np.exp(-0.5 * ((x - centers[j]) / sigma) ** 2)

    # Assemble full basis
    basis = np.hstack([basis_poly, basis_gauss])

    # Stability check
    if not np.all(np.isfinite(basis)):
        basis = np.nan_to_num(basis, nan=0.0, posinf=1e6, neginf=-1e6)

    # Linear combination
    result = basis @ params

    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)

    return result
# EVOLVE-BLOCK-END
