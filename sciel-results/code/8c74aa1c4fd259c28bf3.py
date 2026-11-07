"""RBF interpolant with 10 fixed Gaussian centers in normalized input space."""
# EVOLVE-BLOCK-START
import numpy as np

# Precompute centers in normalized input space
# b: [1, 4.998] -> [0, 1], s: [1, 4.999] -> [0, 1]
# temp: [1.002, 39.98] -> [0, 1], pH: [2, 12] -> [0, 1]
_CENTERS = np.array([
    [0.0, 0.0, 0.0, 0.0],   # all low
    [1.0, 1.0, 1.0, 1.0],   # all high
    [0.0, 0.0, 1.0, 1.0],   # low b,s; high temp,pH
    [1.0, 1.0, 0.0, 0.0],   # high b,s; low temp,pH
    [0.5, 0.5, 0.5, 0.5],   # centroid
    [0.0, 1.0, 0.0, 1.0],   # alternating
    [1.0, 0.0, 1.0, 0.0],   # alternating
    [0.5, 0.0, 0.5, 1.0],   # mid b, low s, mid temp, high pH
    [0.5, 1.0, 0.5, 0.0],   # mid b, high s, mid temp, low pH
    [0.0, 0.5, 1.0, 0.5],   # low b, mid s, high temp, mid pH
])
_WIDTH_SQ = 0.49  # (0.7)^2, moderate width for smooth interpolation


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """E. coli growth rate via RBF interpolation with 10 fitted weights."""
    # Normalize inputs to [0, 1]
    bn = (b - 1.0) / 4.0
    sn = (s - 1.0) / 4.0
    tn = (temp - 1.0) / 39.0
    pn = (pH - 2.0) / 10.0

    # Stack normalized inputs: (n, 4)
    X = np.stack([bn, sn, tn, pn], axis=-1)

    # Compute squared distances to each center: (n, 10)
    # X shape: (n, 4), _CENTERS shape: (10, 4)
    # diff shape: (n, 10, 4)
    diff = X[:, None, :] - _CENTERS[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)

    # Gaussian RBF values: (n, 10)
    rbf = np.exp(-dist_sq / (2.0 * _WIDTH_SQ))

    # Linear combination with fitted weights
    return rbf @ params
# EVOLVE-BLOCK-END
