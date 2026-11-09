"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.

    Strategy: Fragmentation-aware scoring that penalizes bins where the item
    creates moderate leftover fragments (20-50% of remaining space), while
    still preferring tighter fits overall.
    """
    t_start = time.time()
    deadline = t_start + 0.008  # 8ms budget per call with safety margin

    remaining = bins - item
    valid = remaining >= 0

    # Base score: negative remaining capacity (best-fit) — tighter is better.
    score = -remaining.copy()

    # Invalidate bins that cannot hold the item.
    score[~valid] = -np.inf

    if not np.any(valid):
        return score

    post = remaining[valid]  # remaining capacity after placing item
    bins_valid = bins[valid]

    # --- Fragmentation-aware penalty ---
    # The ratio of item to pre-placement remaining capacity indicates how
    # "fragmenting" this placement is.
    # - Very small ratio (<0.2): item is tiny relative to space, fine
    # - Moderate ratio (0.2-0.5): creates a medium-sized leftover that's
    #   hard to fill later — this is the wasteful zone
    # - Large ratio (>0.5): item fills most of the space, leftover is small
    #
    # We compute frag_ratio = item / bins_valid (fraction of bin used)
    # and penalize when the leftover fraction (1 - frag_ratio) is in a
    # moderate range that's hard to utilize.

    # Leftover fraction: what fraction of the bin remains after placement
    leftover_frac = np.where(bins_valid > 1e-9, post / bins_valid, 0.0)

    # Smooth penalty that peaks when leftover_frac is in [0.2, 0.5]
    # Use a smooth "bump" function: penalty peaks at center, tapers off
    # center = 0.35, width = 0.15
    center = 0.35
    width = 0.12
    # Gaussian-like bump: exp(-((x - center) / width)^2)
    bump = np.exp(-((leftover_frac - center) / width) ** 2)
    frag_penalty = 0.4 * bump

    # Additional: penalize very large leftovers (waste of capacity)
    # This is the classic best-fit pressure: large post is bad
    large_leftover_penalty = 0.15 * np.clip(post / 100.0, 0, 1)

    # Bonus for bins already partially filled (encourage consolidation)
    # Using existing bins is better than opening new ones
    util = np.where(bins_valid > 1e-9, item / bins_valid, 0.0)
    # Gentle bonus that saturates
    consolidation_bonus = 0.1 * (1.0 - np.exp(-3.0 * util))

    # Small bonus for moderate utilization to balance load
    moderate_bonus = 0.03 * util * (1.0 - util)

    # Combine all components
    total_adjustment = -frag_penalty - large_leftover_penalty + consolidation_bonus + moderate_bonus

    score[valid] += total_adjustment

    # Deadline guard: if we're running long, fall back to simple best-fit
    if time.time() > deadline:
        scores = np.full_like(bins, -1e9, dtype=float)
        scores[valid] = -remaining[valid]
        return scores

    return score
# EVOLVE-BLOCK-END
