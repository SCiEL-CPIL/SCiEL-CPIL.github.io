import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    t0 = time.time()
    deadline = t0 + 115.0

    remaining = bins - item
    feasible = remaining >= 0
    scores = np.full(len(bins), -np.inf)

    if not np.any(feasible):
        return scores

    rem = remaining[feasible]
    n = len(rem)

    # Ratio of item to remaining capacity before placing
    ratio = item / (rem + 1e-12)

    # Best-fit base: strongly prefer bins with least remaining after placement
    # Using a sharper non-linear transform than sqrt to emphasize tight fits
    base_score = -(rem + 1e-8) ** 0.7

    # Fragmentation penalty via smooth Gaussian bump
    # Penalize ratios in the "wasteful middle" where residual is neither tiny nor large
    center = 0.32
    width = 0.13
    penalty = np.exp(-0.5 * ((ratio - center) / width) ** 2)
    penalty_scale = 1.5
    adjusted = base_score - penalty_scale * penalty

    # Soft-min surrogate: reward being among the best K candidates
    # This smooths the discrete "pick best" signal into a continuous ranking bonus
    K = max(1, int(n * 0.15))
    # Sort descending (highest score first); rank 0 is best
    order = np.argsort(-adjusted)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    # Bonus that decays with rank, emphasizing top-K
    rank_bonus = np.exp(-ranks / (K + 1.0))

    # Combine: primary signal is base_score - penalty, secondary is rank-aware boost
    # The rank bonus is small relative to base_score spread but breaks ties meaningfully
    final_scores = adjusted + 0.05 * rank_bonus

    scores[feasible] = final_scores

    if time.time() > deadline:
        return scores

    return scores
