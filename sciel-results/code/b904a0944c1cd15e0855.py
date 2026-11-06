"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # First-Fit: prefer the first bin that can fit the item.
    # The evaluator provides only valid bins.
    # We assign scores such that the bin with the lowest index gets the highest score.
    indices = np.arange(len(bins))
    return -indices
# EVOLVE-BLOCK-END
