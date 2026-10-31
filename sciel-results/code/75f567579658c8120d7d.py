"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Best-fit: prefer bins that leave the least remaining capacity.
    # We use a slight smoothing to avoid ties and encourage tighter fits.
    # Score is negative of remaining capacity after placing item.
    # Bins with insufficient capacity get a very negative score (effectively excluded).
    remaining = bins - item
    # For bins that can fit the item, remaining >= 0
    # We want to minimize remaining (best fit), so score = -remaining
    # For bins that cannot fit, we give them a very low score
    scores = -remaining
    # Penalize bins that cannot fit the item
    scores[remaining < 0] = -np.inf
    return scores
# EVOLVE-BLOCK-END
