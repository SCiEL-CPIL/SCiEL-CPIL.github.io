"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Score bins: prefer those with enough capacity, using a blend of
    # best-fit (minimize remaining) and worst-fit (maximize remaining)
    # to balance fragmentation vs. utilization for Weibull distributions.
    rem = bins - item
    # Only score valid bins (rem >= 0); invalid bins get -inf
    scores = np.where(rem >= 0, 0.0, -np.inf)
    valid = rem >= 0
    if np.any(valid):
        # Blend: primarily best-fit, with a small worst-fit component
        # to avoid over-fragmenting bins with very little space left
        best_fit_score = -rem[valid]
        worst_fit_score = rem[valid]
        # Normalize to [0, 1] range for blending
        min_rem = rem[valid].min()
        max_rem = rem[valid].max()
        if max_rem > min_rem:
            norm_rem = (rem[valid] - min_rem) / (max_rem - min_rem)
        else:
            norm_rem = np.zeros_like(rem[valid])
        # 80% best-fit, 20% worst-fit
        blended = 0.8 * best_fit_score + 0.2 * worst_fit_score
        scores[valid] = blended
    return scores
# EVOLVE-BLOCK-END
