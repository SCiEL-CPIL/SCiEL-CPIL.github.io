import numpy as np
import time

# Consolidation bias: positive weight on current bin fill to prefer
# packing into already-used bins (reduces total bin count vs. L1 bound).
_BETA = 0.15
_INFEASIBLE = -1e9

# Deadline guard: track start time to ensure we return within 120s limit.
# We add a safety margin of 5 seconds.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # 5s safety margin before 120s kill


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Score each bin with enough remaining capacity; higher = preferred.

    Primary term: tightest fit (minimize remaining space after placement).
    Secondary term: consolidation bias toward already-used bins.
    Tertiary term: quadratic tightening bonus for very tight fits.
    """
    # Check deadline guard
    if time.time() > _DEADLINE:
        # Return a safe default: prefer tightest fit with no extra computation
        bins = np.asarray(bins, dtype=float)
        remaining = bins - item
        feasible = remaining >= 0.0
        score = -remaining
        score = np.where(feasible, score, _INFEASIBLE)
        return score

    bins = np.asarray(bins, dtype=float)
    remaining = bins - item
    feasible = remaining >= 0.0

    # Tightest fit: prefer smallest remaining (negative remaining).
    score = -remaining

    # Consolidation bias: prefer bins that already hold more.
    score = score + _BETA * bins

    # Quadratic tightening: reward bins where remaining is very small relative to item.
    # This makes the preference for tight fits super-linear, strongly favoring
    # bins where the item almost exactly fills the remaining space.
    # Use a scaled quadratic term to avoid numerical issues.
    if item > 0:
        tightness = np.where(feasible, remaining, 1.0)
        # Scale by item size to keep bonus meaningful across different item sizes
        bonus = -tightness ** 2 / (item * 10.0 + 1.0)
        score = score + bonus

    # Mask infeasible bins so they are never selected.
    score = np.where(feasible, score, _INFEASIBLE)

    return score
