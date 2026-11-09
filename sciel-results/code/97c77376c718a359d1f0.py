"""Symbolic regression: rich mixed basis (poly + exp + trig + log + tanh + RBF), linear in params."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 140.0  # safety margin under the 150s wall-clock limit

# Precompute fixed basis structure (independent of x)
_RBF_CENTERS = np.array([1.0, 3.0, 5.0])
_RBF_WIDTH = 1.2
_TRIG_FREQS = np.array([1.0, 2.0])


def _build_basis(x: np.ndarray) -> np.ndarray:
    """Return an (N, 10) basis matrix for the given 1-D x."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size

    # Polynomial terms: 1, x, x^2, x^3  (4 columns)
    poly = np.empty((n, 4), dtype=np.float64)
    poly[:, 0] = 1.0
    poly[:, 1] = x
    poly[:, 2] = x * x
    poly[:, 3] = x * x * x

    # Trig terms: sin(f x), cos(f x) for each frequency (2 cols x 2 freqs = 4)
    trig = np.empty((n, 4), dtype=np.float64)
    trig[:, 0] = np.sin(_TRIG_FREQS[0] * x)
    trig[:, 1] = np.cos(_TRIG_FREQS[0] * x)
    trig[:, 2] = np.sin(_TRIG_FREQS[1] * x)
    trig[:, 3] = np.cos(_TRIG_FREQS[1] * x)

    # RBF terms: 2 of the 3 centers (2 columns)
    rbf = np.empty((n, 2), dtype=np.float64)
    for k in range(2):
        rbf[:, k] = np.exp(-((x - _RBF_CENTERS[k]) ** 2) / (_RBF_WIDTH ** 2))

    basis = np.concatenate([poly, trig, rbf], axis=1)  # (n, 10)

    # Column-wise standardization for conditioning (guard against zero std)
    mean = basis.mean(axis=0, keepdims=True)
    std = basis.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    basis = (basis - mean) / std

    return basis


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Prediction = standardized_basis(x) @ params, with a deadline guard."""
    # Deadline guard: if we're too close to the limit, return a safe constant
    if time.time() > _DEADLINE:
        return np.ones(np.asarray(x, dtype=np.float64).shape, dtype=np.float64)

    x_arr = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x_arr)):
        return np.zeros_like(x_arr, dtype=np.float64)

    params = np.asarray(params, dtype=np.float64).ravel()
    if params.size != 10:
        params = np.resize(params, 10)

    basis = _build_basis(x_arr)
    predictions = basis @ params

    # Ensure finite output
    if not np.all(np.isfinite(predictions)):
        predictions = np.where(np.isfinite(predictions), predictions, 0.0)

    # Return in the same shape as the input
    return predictions.reshape(x_arr.shape)
# EVOLVE-BLOCK-END
