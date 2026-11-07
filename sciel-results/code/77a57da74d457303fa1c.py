"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    dm = np.asarray(distance_matrix, dtype=float)
    if dm.ndim != 2 or dm.shape[0] != dm.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    d = np.maximum(dm, 1e-9)

    # Use a stronger inverse power to more aggressively favor short edges.
    # For 50 random points in unit square, nearest neighbors are ~0.01-0.05,
    # far edges ~0.5-1.0. A higher power creates sharper contrast.
    h = 1.0 / (d ** 2.0)

    # Normalize to mean 1.0 so heuristic scale matches initial pheromone scale.
    mean_h = np.mean(h)
    if not np.isfinite(mean_h) or mean_h <= 0.0:
        h = np.ones_like(dm, dtype=float)
    else:
        h = h / mean_h

    # Contract requires positive entries; non-positive becomes 1e-9.
    h = np.where(np.isfinite(h) & (h > 0.0), h, 1e-9)
    return h
# EVOLVE-BLOCK-END
