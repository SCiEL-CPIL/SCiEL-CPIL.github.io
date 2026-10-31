"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 120.0
_start_time = time.time()

# Pre-compute Weibull-based expected gap statistics.
# Weibull with shape k~2, scale lambda~12 gives mean ~10.7, mode ~8.5.
# For capacity 100, the "optimal" residual gap after packing should be
# small but non-zero. We model the target gap distribution as a Weibull
# and score bins by how close the resulting gap is to the target mode.
#
# The key insight: instead of minimizing gap locally (best-fit) or maximizing
# it (worst-fit), we want the gap to be in the "sweet spot" predicted by
# the item size distribution. For Weibull(k=2, lambda=12):
#   mode = lambda * (k-1)^(1/k) = 12 * 1^(0.5) = 12
#   mean = lambda * gamma(1+1/k) = 12 * gamma(1.5) ≈ 12 * 0.8862 ≈ 10.63
#
# We use a quadratic penalty around the target gap, which is equivalent
# to minimizing KL divergence between a point mass at the resulting gap
# and the target Weibull PDF (up to a constant).

CAPACITY = 100.0
# Target gap: use the mode of Weibull(k=2, lambda=12) ≈ 12.0
# This is the gap size most likely to be "useful" for future items.
TARGET_GAP = 12.0
# Width of the tolerance: how sharply we penalize deviation from target.
# Smaller sigma = sharper peak = more selective.
SIGMA = 8.0


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Safety fallback: best fit
        remaining = bins - item
        feasible = remaining >= 0
        return np.where(feasible, -remaining, -np.inf)

    remaining = bins - item
    feasible = remaining >= 0

    # KL-divergence-like score: -0.5 * ((gap - target) / sigma)^2
    # This is the negative log-likelihood of the gap under a Gaussian
    # approximation to the Weibull PDF centered at the mode.
    # Higher score = gap closer to target = statistically "equilibrated".
    gap_deviation = remaining - TARGET_GAP
    kl_score = -0.5 * (gap_deviation / SIGMA) ** 2

    # Additional component: for very small gaps (< 5), the bin is nearly full
    # and this is good (saves the bin). For very large gaps (> 40), the bin
    # is mostly empty and wasting space. The KL term handles this naturally,
    # but we add a slight boost for near-full bins to encourage consolidation.
    near_full_bonus = np.where(remaining < 5.0, (5.0 - remaining) * 0.05, 0.0)

    scores = np.where(
        feasible,
        kl_score + near_full_bonus,
        -np.inf
    )

    return scores
# EVOLVE-BLOCK-END
