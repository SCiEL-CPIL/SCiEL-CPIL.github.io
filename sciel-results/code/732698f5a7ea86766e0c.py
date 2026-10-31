"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from time import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.

    Strategy: Combine best-fit (minimize leftover) with a global
    variance-aware penalty to reduce fragmentation across bins.
    """
    t0 = time()
    deadline = 140.0  # safety margin before 150s kill

    remaining = bins - item
    # Only bins with enough capacity are eligible
    eligible = remaining >= 0

    # Best-fit base score: prefer bins where leftover is smallest
    # We use negative leftover so higher score = better fit
    scores = np.where(eligible, -remaining, -np.inf)

    # Add a small global-awareness term:
    # If the remaining capacities are highly variable, favor bins
    # that bring the system closer to the mean (reduce variance)
    eligible_remaining = remaining[eligible]
    if len(eligible_remaining) > 1:
        mean_rem = np.mean(eligible_remaining)
        # Penalty for bins that are far from the mean remaining
        # This encourages a more uniform distribution of leftover space
        variance_penalty = 0.01 * (remaining - mean_rem) ** 2
        scores = np.where(eligible, scores - variance_penalty, -np.inf)

    # Check time budget
    if time() - t0 > deadline:
        # Fallback: simple best-fit
        return -remaining

    return scores
# EVOLVE-BLOCK-END
