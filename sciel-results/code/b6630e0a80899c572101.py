"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

_INFEASIBLE = -1e18


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Score each bin with enough remaining capacity; higher = preferred.

    Strategy: tightest-fit (minimize remaining after placement) combined
    with a strong consolidation bias that penalizes low-fill bins,
    encouraging the algorithm to fill bins before opening new ones.
    """
    bins = np.asarray(bins, dtype=float)
    remaining = bins - item
    feasible = remaining >= 0.0

    if not np.any(feasible):
        return np.full_like(bins, _INFEASIBLE, dtype=float)

    # Primary: tightest fit — prefer smallest remaining space.
    score = -remaining.copy()

    # Consolidation: strongly prefer bins that are already substantially filled.
    # Use a quadratic fill preference: bins with higher current fill get a
    # bonus that grows with fill level, so half-full bins are strongly preferred
    # over nearly-empty ones, but the tightest-fit term still dominates among
    # similarly-filled feasible bins.
    fill = 1.0 - remaining  # fraction of bin capacity used (if capacity=100, this is bins/100 scaled)
    # Actually bins is remaining capacity, so fill_fraction = 1 - remaining/C.
    # We don't know C explicitly, but bins values are in [0, 100].
    # Use bins directly: higher bins = more remaining = less filled.
    # So consolidation bonus should prefer LOWER remaining (more filled).
    # The -remaining term already does this. Add a quadratic boost for very tight fits.

    # Quadratic tightening: reward bins where remaining is very small relative to item.
    # This makes the preference for tight fits super-linear, strongly favoring
    # bins where the item almost exactly fills the remaining space.
    tightness = np.where(feasible, remaining, 1.0)
    # Bonus: -tightness^2 / (item + 1) to scale with item size
    bonus = -tightness ** 2 / (item + 1.0)

    score = score + bonus

    # Additional consolidation: small linear bonus for bins with less remaining
    # (already captured by -remaining, but add a mild extra push)
    score = score + 0.1 * (-remaining)

    # Mask infeasible bins
    score = np.where(feasible, score, _INFEASIBLE)

    return score
# EVOLVE-BLOCK-END
