"""Symbolic regression with 2 inputs using fixed-geometry RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _halton2d(n, base1=2, base2=3):
    """Generate n 2D Halton sequence points in [0,1]^2."""
    def halton1d(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    points = np.zeros((n, 2))
    for i in range(1, n + 1):
        points[i - 1, 0] = halton1d(i, base1)
        points[i - 1, 1] = halton1d(i, base2)
    return points


def _build_rbf_basis(x, y, n_centers=10):
    """
    Build RBF basis matrix with fixed centers (Halton sequence) and widths.
    Returns (B, centers, widths) where B has shape (n_points, n_centers).
    """
    n_points = len(x)
    # Generate centers using Halton sequence mapped to [-1, 1]
    halton = _halton2d(n_centers)
    centers = 2.0 * halton - 1.0  # map [0,1] -> [-1,1]

    # Compute distances from each point to each center
    # centers: (n_centers, 2), x,y: (n_points,)
    dx = x[:, np.newaxis] - centers[:, 0][np.newaxis, :]  # (n_points, n_centers)
    dy = y[:, np.newaxis] - centers[:, 1][np.newaxis, :]  # (n_points, n_centers)
    dist_sq = dx**2 + dy**2  # (n_points, n_centers)

    # Choose width: use median distance between centers for good conditioning
    # Compute pairwise distances between centers
    cx = centers[:, 0]
    cy = centers[:, 1]
    pairwise_dx = cx[:, np.newaxis] - cx[np.newaxis, :]
    pairwise_dy = cy[:, np.newaxis] - cy[np.newaxis, :]
    pairwise_dist_sq = pairwise_dx**2 + pairwise_dy**2
    # Get median of non-zero pairwise distances
    mask = ~np.eye(n_centers, dtype=bool)
    nonzero_dists = np.sqrt(pairwise_dist_sq[mask])
    median_dist = np.median(nonzero_dists) if len(nonzero_dists) > 0 else 1.0

    # Width parameter: sigma = median_dist / sqrt(2) for reasonable spread
    sigma = median_dist / np.sqrt(2.0)

    # RBF: exp(-dist_sq / (2 * sigma^2))
    B = np.exp(-dist_sq / (2.0 * sigma**2))

    return B, centers, sigma


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF-based model with fixed geometry.
    params: array of length 10, interpreted as RBF amplitudes.
    """
    n_points = len(x)
    n_centers = 10

    # Build the fixed RBF basis
    B, centers, sigma = _build_rbf_basis(x, y, n_centers)

    # params are the amplitudes for each RBF
    amplitudes = params[:n_centers]

    # Prediction: sum of amplitude_i * rbf_i
    pred = B @ amplitudes

    return pred
# EVOLVE-BLOCK-END
