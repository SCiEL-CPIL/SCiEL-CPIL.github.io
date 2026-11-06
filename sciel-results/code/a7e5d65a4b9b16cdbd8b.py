"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Worst-Fit: prefer bins that leave the most remaining capacity.
    # This spreads small items, potentially preserving space in bins for larger future items.
    scores = np.full_like(bins, -np.inf, dtype=float)
    valid_mask = bins >= item
    scores[valid_mask] = bins[valid_mask]
    return scores
# EVOLVE-BLOCK-END
