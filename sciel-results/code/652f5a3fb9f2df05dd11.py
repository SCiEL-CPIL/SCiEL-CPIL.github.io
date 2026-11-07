"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    """
    Return an item-by-item matrix of attractiveness for placing item j right
    after item i in the same bin. Larger is more attractive.

    Strategy:
      - If items i and j don't fit together, attractiveness is 1e-9.
      - If they do fit, we want to reward pairs that fill the bin tightly
        (i.e., demands[i] + demands[j] close to capacity) because tight
        packing reduces the number of bins needed.
      - We use a smooth surrogate: (pair_sum / capacity)^2 which strongly
        rewards tight pairs. We also add a small bonus for complementary sizes
        (one large, one small) which can help balance bin loads.
      - The final heuristic is normalized to be positive and finite.
    """
    d = np.asarray(demands, dtype=np.float64)
    n = len(d)
    pair = d[:, None] + d[None, :]
    fits = pair <= capacity

    # Base attractiveness: squared ratio rewards tight packing
    ratio = np.where(fits, pair / capacity, 0.0)
    base = ratio ** 2

    # Complementarity bonus: reward pairs where one item is large and the other
    # is small, which helps balance bin loads. We use (1 - |d_i - d_j| / capacity)
    # as a smooth measure of complementarity.
    diff = np.abs(d[:, None] - d[None, :])
    complement = np.where(fits, 1.0 - diff / capacity, 0.0)

    # Combine: primary weight on tight packing, secondary on complementarity
    heuristic = 0.7 * base + 0.3 * complement

    # Ensure non-positive values become 1e-9
    heuristic = np.where(fits & (heuristic > 0), heuristic, 1e-9)

    # Ensure all values are finite
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
