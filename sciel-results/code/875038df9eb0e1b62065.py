"""Item-pair matrix heuristic for ACO offline bin packing."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    """
    Compute an item-by-item attractiveness matrix for placing item j right after
    item i in the same bin. Larger values mean more attractive.

    Strategy: strongly reward tight fits (small remaining space) using an inverse
    relationship, with a sharp decay scale. Non-fitting pairs get a floor of 1e-9.
    """
    d = np.asarray(demands, dtype=float)
    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    cap = float(capacity)

    # Pairwise sums
    pair = d[:, None] + d[None, :]

    # Which pairs fit in a bin of given capacity
    fits = pair <= cap

    # Remaining space if i and j are placed together
    remaining = cap - pair

    # We want to reward tight fits. Use 1 / (1 + (remaining / scale)^2)
    # This gives a smooth, bounded, positive value that decays as remaining grows.
    # A smaller 'scale' makes the decay sharper (stronger preference for tight fits).
    #
    # With capacity=150 and demands in [20, 99], remaining ranges from 0 to ~110.
    # scale=15 gives:
    #   remaining=0   -> 1.0
    #   remaining=15  -> 0.5
    #   remaining=30  -> 1/5 = 0.2
    #   remaining=60  -> 1/17 ≈ 0.059
    #   remaining=110 -> 1/(1+54.4) ≈ 0.018
    #
    # This is sharper than the seed's scale=10 (which used remaining^2/100, i.e.
    # effectively scale=10), but the squared form here is 1/(1+(r/s)^2) which is
    # equivalent to 1/(1 + r^2/s^2). So scale=10 here is the same as the seed.
    # Let's try a slightly sharper scale to push ants toward tighter packing.
    
    scale = 12.0
    denom = 1.0 + (remaining / scale) ** 2

    # Build the result matrix
    h = np.full((n, n), 1e-9, dtype=np.float64)
    
    # For fitting pairs, compute the attractiveness
    h[fits] = 1.0 / denom[fits]

    # Ensure no non-finite or non-positive values leak through
    h[~np.isfinite(h)] = 1e-9
    h[h <= 0] = 1e-9

    return h
# EVOLVE-BLOCK-END
