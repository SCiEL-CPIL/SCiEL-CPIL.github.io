"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    pair = demands[:, None] + demands[None, :]
    mask = pair <= capacity

    # Tightness: how close the pair sum is to capacity (1.0 = exactly full)
    tightness = np.where(mask, pair / capacity, 0.0)

    # Gaussian-like bump centered at tightness=1.0 with moderate width.
    # This strongly favors pairs that fill the bin almost exactly, and
    # smoothly decays for looser pairs.  Width sigma=0.08 gives a sharp
    # peak: a pair at 92% capacity gets ~0.37 of the peak, at 80% ~0.05.
    sigma = 0.08
    bump = np.exp(-0.5 * ((tightness - 1.0) / sigma) ** 2)

    # Complementarity: items that are both "medium" (around capacity/2)
    # tend to pair well; add a mild boost when both sizes are near the
    # midpoint of the feasible range.  This breaks ties among several
    # tight pairs by preferring balanced ones.
    mid = capacity / 2.0
    complement = np.exp(-((demands[:, None] - mid) ** 2 + (demands[None, :] - mid) ** 2) / (2 * (capacity * 0.25) ** 2))

    # Combine: primary signal is the tightness bump, secondary is complementarity.
    heuristic = np.where(mask, bump * (0.8 + 0.2 * complement), 1e-9)

    # Ensure no zero or negative entries (mask already handles infeasible)
    heuristic = np.maximum(heuristic, 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
