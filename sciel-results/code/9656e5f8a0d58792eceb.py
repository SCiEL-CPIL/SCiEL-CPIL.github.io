"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Best-fit: prefer bins that leave the least remaining capacity.
    # Only consider bins with enough remaining capacity.
    remaining = bins - item
    # Set score to -inf for bins that can't fit the item
    scores = np.where(remaining >= 0, -remaining, -np.inf)
    return scores
# EVOLVE-BLOCK-END
