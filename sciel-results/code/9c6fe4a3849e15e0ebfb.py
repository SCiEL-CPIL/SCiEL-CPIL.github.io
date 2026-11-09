"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Worst-Fit: prefer bins with the most remaining capacity after placing item.
    # This preserves high-capacity bins for future large items, reducing fragmentation.
    remaining = bins - item
    # Only score bins with enough remaining capacity (non-negative remaining)
    scores = np.where(remaining >= 0, remaining, -np.inf)
    return scores
# EVOLVE-BLOCK-END
