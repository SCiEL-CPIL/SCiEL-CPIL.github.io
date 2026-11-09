"""Item-pair attractiveness heuristic for ACO bin packing."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    d = np.asarray(demands, dtype=float)
    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    cap = float(capacity)
    pair = d[:, None] + d[None, :]
    fits = pair <= cap
    remaining = cap - pair  # space left if i and j share a bin

    # Complementary pairing: reward pairs that fill the bin tightly.
    # Prefer pairs whose combined size is close to capacity (small waste),
    # and penalise pairs that leave a lot of unused space.
    h = np.zeros((n, n), dtype=float)
    h[fits] = remaining[fits] ** (-1.0)          # 1/remaining -> bigger for tight fits
    h[~fits] = 1e-9                              # infeasible / non-positive -> floor

    # Guard against any non-finite values (e.g. remaining == 0).
    h[~np.isfinite(h)] = 1e-9
    h[h <= 0] = 1e-9

    return h
# EVOLVE-BLOCK-END
