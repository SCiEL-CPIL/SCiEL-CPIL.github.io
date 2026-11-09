import numpy as np

# Consolidation bias: positive weight on current bin fill to prefer
# packing into already-used bins (reduces total bin count vs. L1 bound).
_BETA = 0.15
_GAMMA = 0.01
_INFEASIBLE = -1e9


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Score each bin with enough remaining capacity; higher = preferred.

    Primary term: tightest fit (minimize remaining space after placement).
    Secondary term: consolidation bias toward already-used bins.
    """
    bins = np.asarray(bins, dtype=float)
    remaining = bins - item
    feasible = remaining >= 0.0

    # Tightest fit: prefer smallest remaining (negative remaining).
    score = -remaining

    # Consolidation bias: prefer bins that already hold more.
    score = score + _BETA * bins

    # Index penalty: prefer earlier bins (assuming bins are ordered by creation)
    # This helps consolidate into older bins, mimicking First Fit behavior.
    if len(bins) > 0:
        indices = np.arange(len(bins))
        score = score - _GAMMA * indices

    # Mask infeasible bins so they are never selected.
    score = np.where(feasible, score, _INFEASIBLE)

    return score
