"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    """
    Return an item-by-item attractiveness matrix for ACO bin packing.
    Heuristic favors pairs that leave useful remaining capacity for future items.
    """
    d = np.asarray(demands, dtype=np.float64)
    n = len(d)
    cap = float(capacity)

    # Pairwise sum and remaining capacity
    pair_sum = d[:, None] + d[None, :]
    remaining = cap - pair_sum

    # Only valid if both items fit individually and together
    fits = (d[:, None] <= cap) & (d[None, :] <= cap) & (remaining >= 0)

    # Attractiveness: prefer pairs that leave moderate remaining capacity
    # (enough for at least one more small item, but not so much that we waste space)
    # Also prefer pairs where both items are reasonably large (fills bins efficiently)
    # Use a smooth scoring:
    #   - reward for filling the bin (small remaining)
    #   - reward for the pair sum being close to capacity
    #   - small bonus for items that can still fit another item after (flexibility)

    # Normalize demands to [0,1] for scaling
    d_min = d.min()
    d_max = d.max()
    d_range = d_max - d_min
    if d_range < 1e-9:
        d_norm = np.ones_like(d) * 0.5
    else:
        d_norm = (d - d_min) / d_range

    # Fill ratio: how full the bin is after placing both items
    fill_ratio = pair_sum / cap  # in [0, 2] potentially, but capped by fits

    # Score components:
    # 1. Fill score: prefer fill_ratio close to 1.0 (full bin)
    fill_score = 1.0 - np.abs(fill_ratio - 1.0)  # peaks at 1.0

    # 2. Remaining capacity utility: remaining should be >= min demand (so another item can fit)
    #    but not too large (waste). Smooth penalty for too-much remaining.
    min_d = d.min()
    # Utility: high when remaining is in [min_d, 0.3*cap], drops off outside
    rem_util = np.where(
        remaining >= min_d,
        np.exp(-np.maximum(remaining - min_d, 0) / (0.2 * cap)),
        0.0
    )

    # 3. Item size synergy: prefer pairing items that are both substantial
    #    (avoids putting tiny items with tiny items, wasting space)
    size_product = d_norm[:, None] * d_norm[None, :]
    size_score = 0.3 + 0.7 * size_product  # [0.3, 1.0]

    # 4. Complementarity: prefer pairs where one is large and one is medium (fills gaps)
    #    Actually, for bin packing, pairing similar sizes or complementary sizes both work.
    #    Let's use: prefer pairs where the sum is close to capacity (already in fill_score)

    # Combine scores
    # We want the heuristic to be > 0 for valid pairs, and very small (1e-9) for invalid
    combined = fill_score * rem_util * size_score

    # Ensure positive values for valid pairs
    # Scale to a reasonable range
    max_val = combined[fits].max() if np.any(fits) else 1.0
    if max_val < 1e-12:
        max_val = 1.0
    combined = combined / max_val

    # Apply mask: invalid pairs get 1e-9
    result = np.where(fits, combined, 1e-9)

    # Ensure no non-finite values
    result = np.where(np.isfinite(result), result, 1e-9)

    # Ensure all values are positive
    result = np.maximum(result, 1e-9)

    return result
# EVOLVE-BLOCK-END
