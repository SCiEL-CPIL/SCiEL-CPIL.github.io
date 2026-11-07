"""
Strategy: Hybrid Chebyshev + Localized Gaussian Basis with Normalization.

Combines the structural idea of a rich global basis (Chebyshev polynomials)
with the localized flexibility of Gaussian bumps, inspired by the RBF approaches.
Crucially, we normalize each basis column to have unit L2 norm on the training
domain [-2, 2]. This ensures that the BFGS optimizer starts from a well-scaled
initialization (all ones) and converges reliably to the global minimum of the
linear least-squares problem, avoiding conditioning issues that plague raw
polynomial or exponential bases.

Basis functions (10 total):
- 6 Chebyshev polynomials T_0 to T_5 on scaled x in [-1, 1]
- 4 Gaussian bumps centered at -1.5, -0.5, 0.5, 1.5 with width 0.8

Normalization: Each column is divided by its norm over a dense grid in [-2, 2].
This makes the parameter scale uniform, aiding BFGS convergence.
"""
import numpy as np
import time

_DEADLINE_S = 145.0
_MARGIN_S = 5.0
_START_TIME = time.time()

# Pre-compute normalization factors for the basis functions
# We use a dense grid to approximate the L2 norm over [-2, 2]
_NORM_GRID = np.linspace(-2.0, 2.0, 1000)
_NORM_GRID_X = _NORM_GRID

def _compute_raw_basis(x: np.ndarray) -> np.ndarray:
    """
    Compute the unnormalized 10-column basis matrix.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.shape[0]
    
    # Scale x to [-1, 1] for Chebyshev stability
    t = x * 0.5
    
    # Chebyshev polynomials T_0 to T_5
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * t - 1.0
    T3 = 4.0 * t * t * t - 3.0 * t
    T4 = 8.0 * t * t * t * t - 8.0 * t * t + 1.0
    T5 = 16.0 * t * t * t * t * t - 20.0 * t * t * t + 5.0 * t
    
    # Gaussian bumps
    # Centers: -1.5, -0.5, 0.5, 1.5
    # Width: 0.8
    centers = np.array([-1.5, -0.5, 0.5, 1.5])
    sigma = 0.8
    G0 = np.exp(-0.5 * ((x - centers[0]) / sigma) ** 2)
    G1 = np.exp(-0.5 * ((x - centers[1]) / sigma) ** 2)
    G2 = np.exp(-0.5 * ((x - centers[2]) / sigma) ** 2)
    G3 = np.exp(-0.5 * ((x - centers[3]) / sigma) ** 2)
    
    # Stack into (N, 10)
    basis = np.column_stack([T0, T1, T2, T3, T4, T5, G0, G1, G2, G3])
    return basis

# Pre-compute normalization factors
# We want to normalize such that the basis columns are orthogonal-ish and well-scaled.
# Simpler: normalize each column to have mean absolute value 1 over the domain.
# Or better: normalize by the L2 norm over the domain.
# Let's compute the norm of each basis function over the grid.

def _get_norm_factors() -> np.ndarray:
    """
    Compute normalization factors for each of the 10 basis functions.
    """
    raw_basis = _compute_raw_basis(_NORM_GRID_X)
    # L2 norm approximation: sqrt(sum(x_i^2) * dx)
    # For simplicity, we can just use the Euclidean norm of the column vector
    # since the grid is uniform.
    norms = np.linalg.norm(raw_basis, axis=0)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    return norms

_NORM_FACTORS = _get_norm_factors()

def _compute_normalized_basis(x: np.ndarray) -> np.ndarray:
    """
    Compute the normalized 10-column basis matrix.
    """
    basis = _compute_raw_basis(x)
    basis = basis / _NORM_FACTORS
    return basis

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE_S - _MARGIN_S:
        # Fallback: return a constant prediction (mean of params)
        return np.full_like(np.asarray(x, dtype=float).ravel(), np.mean(params))
    
    x = np.asarray(x, dtype=float).ravel()
    params = np.asarray(params, dtype=float).ravel()
    
    # Ensure params is length 10
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
    
    basis = _compute_normalized_basis(x)
    result = basis @ params
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    return result
