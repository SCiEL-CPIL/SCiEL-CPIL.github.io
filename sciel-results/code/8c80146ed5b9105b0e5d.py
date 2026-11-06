"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Worst-fit: prefer bins that leave the most remaining capacity.
    # This preserves tightly-fitting bins for future items.
    return bins - item
# EVOLVE-BLOCK-END
