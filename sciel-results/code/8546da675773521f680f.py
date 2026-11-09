"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Score each bin with enough remaining capacity.

    Uses a structural change: instead of a fixed best-fit + small bonus,
    we use a smooth sigmoid-based cost landscape that penalizes moderate
    fragments heavily while being lenient toward bins that are either
    nearly full after placement (good) or have large remaining capacity
    (flexible for future large items). This creates a bimodal preference
    structure that avoids the pathological fragmentation of pure best-fit.

    The score is designed so that higher scores are preferred (the item
    goes to the highest-scoring bin).

    We return a vector of scores; bins that cannot hold the item get -inf.
    """
    start = time.perf_counter()
    deadline = start + 115.0  # safety margin

    remaining = bins - item
    feasible = remaining >= 0

    if not np.any(feasible):
        # No feasible bin; return all -inf so no bin is chosen
        return np.full_like(bins, -np.inf, dtype=float)

    scores = np.full_like(bins, -np.inf, dtype=float)
    feasible_indices = np.where(feasible)[0]
    rem = remaining[feasible_indices]
    bin_vals = bins[feasible_indices]

    # Check deadline early
    if time.perf_counter() > deadline:
        return scores

    # Structural change: use a sigmoid-based smooth cost landscape.
    # 
    # Key insight: pure best-fit (minimize remaining) creates many small
    # fragments. A bimodal preference for residuals near 0 (nearly full)
    # OR near capacity (very flexible) outperforms pure best-fit.
    #
    # We model the "quality" of a placement as a function of the remaining
    # capacity after placing the item. We want:
    #   - High score when remaining is very small (bin nearly full)
    #   - High score when remaining is large (bin still very flexible)
    #   - Low score for moderate remaining (wasteful fragment)
    #
    # Use two sigmoid terms:
    #   term1: favors small remaining (best-fit-like)
    #   term2: favors large remaining (flexibility)
    #
    # The combination creates a bimodal landscape.

    # Normalize remaining to [0, 1] relative to bin capacity
    # This makes the scoring scale-invariant
    norm_rem = rem / bin_vals  # fraction of bin still available after placement

    # term1: sigmoid that gives high score when norm_rem is small
    # sigma1 = 0.15, centered at 0.3
    # When norm_rem < 0.3, term1 is high; when norm_rem > 0.3, term1 drops
    center1 = 0.25
    width1 = 0.12
    term1 = 1.0 / (1.0 + np.exp((norm_rem - center1) / width1))

    # term2: sigmoid that gives high score when norm_rem is large
    # sigma2 = 0.2, centered at 0.7
    # When norm_rem > 0.7, term2 is high; when norm_rem < 0.7, term2 drops
    center2 = 0.75
    width2 = 0.15
    term2 = 1.0 / (1.0 + np.exp(-(norm_rem - center2) / width2))

    # Combine: weighted sum of the two terms
    # Weight term1 slightly more heavily since best-fit is generally good
    w1 = 1.0
    w2 = 0.6

    scores[feasible_indices] = w1 * term1 + w2 * term2

    # Check deadline
    if time.perf_counter() > deadline:
        return scores

    return scores
# EVOLVE-BLOCK-END
