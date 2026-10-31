"""Rewritten online bin-packing priority: probabilistic best-fit with polarization."""
# EVOLVE-BLOCK-START
import numpy as np

# Assumed bin capacity for Weibull benchmark sequences
_CAP = 100.0

# Expected next-item size proxy (tuned to Weibull-like item distributions)
# We aim to leave a gap around this value when feasible
_EXPECTED_NEXT = 15.0

# Sharpness of probabilistic transition for the expected-next-item gap
_SIGMA = 8.0

# Weights balancing immediate fit vs. future flexibility
_W_BESTFIT = 1.0
_W_EXPECTED = 1.2
_W_POL = 0.35


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin that can fit the item.
    The item is placed in the bin with the highest score.

    Scoring design:
      - Base: prefer smaller remaining gaps (best-fit) to reduce fragmentation.
      - Probabilistic term: reward gaps near the expected next item size,
        using a smooth exponential kernel to avoid creating unusable slivers.
      - Polarization term: penalize bins that are near half-full, since they
        have the least flexibility for future items of varying sizes.
    """
    n = bins.size
    scores = np.full(n, -np.inf, dtype=np.float64)

    if n == 0:
        return scores

    remaining = bins - item
    feasible = remaining >= 0.0

    if not np.any(feasible):
        return scores

    rem = remaining[feasible]

    # 1) Best-fit component: smaller gap is better
    bestfit = -rem * _W_BESTFIT

    # 2) Probabilistic kernel around expected next item
    #    Score peaks when rem == _EXPECTED_NEXT
    z = (rem - _EXPECTED_NEXT) / _SIGMA
    z = np.clip(z, -8.0, 8.0)
    kernel = np.exp(-z * z) * _W_EXPECTED

    # 3) Polarization penalty: minimize distance to either empty or full
    #    min(rem, CAP - rem) is small when near empty or full (good),
    #    large when near half-full (bad). We penalize larger values.
    pol = np.minimum(rem, _CAP - rem)
    pol_penalty = -pol * _W_POL

    combined = bestfit + kernel + pol_penalty

    scores[feasible] = combined
    return scores
# EVOLVE-BLOCK-END
