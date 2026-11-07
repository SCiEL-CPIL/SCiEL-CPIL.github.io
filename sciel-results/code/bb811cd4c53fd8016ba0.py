"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    prizes, weights: shape (5, 100), uniform [0,1].
    Returns: array of 100 attractiveness scores (larger = more attractive).
    
    Strategy: Greedy-packing feasibility score.
    - Simulate a greedy knapsack (sort by prize / sum(weights)) with capacity 25 per constraint.
    - Items that fit in the greedy packing get a high base score proportional to their prize.
    - Items that don't fit get a lower score based on their "near-miss" potential.
    - This directly mirrors what the ACO ants will do, focusing pheromone on truly packable items.
    """
    t0 = time.time()
    DEADLINE = 320.0  # safety margin before 330s kill

    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    n_constraints, n_items = weights.shape
    capacity = 25.0

    # Total weight per item across all constraints
    total_w = weights.sum(axis=0)  # shape (100,)
    total_p = prizes.sum(axis=0)   # shape (100,)

    # Efficiency ratio for sorting
    eff = total_p / (total_w + 1e-12)
    order = np.argsort(-eff)  # descending efficiency

    # Greedy packing simulation
    remaining = np.full(n_constraints, capacity, dtype=np.float64)
    packed = np.zeros(n_items, dtype=bool)
    packed_indices = []

    for idx in order:
        if time.time() - t0 > DEADLINE:
            break
        # Check if item fits in all constraints
        if np.all(weights[:, idx] <= remaining):
            remaining -= weights[:, idx]
            packed[idx] = True
            packed_indices.append(idx)

    # Base heuristic: prize-weighted score
    # Packed items: high score proportional to their prize (they're proven useful)
    # Unpacked items: lower score, but still give them a chance based on efficiency
    scores = np.zeros(n_items, dtype=np.float64)

    # For packed items: score = sum of prizes (they contribute directly)
    # Use a power to amplify differences
    packed_mask = packed
    scores[packed_mask] = total_p[packed_mask] ** 1.3

    # For unpacked items: give them a score based on efficiency but dampened
    # They might still be useful in different combinations
    unpacked_mask = ~packed_mask
    if np.any(unpacked_mask):
        # Scale by their efficiency relative to the best packed item
        max_packed_eff = eff[packed_mask].max() if np.any(packed_mask) else 1.0
        rel_eff = eff[unpacked_mask] / (max_packed_eff + 1e-12)
        scores[unpacked_mask] = total_p[unpacked_mask] * rel_eff ** 0.8 * 0.7

    # Additional refinement: items that are "bottleneck" items (consume a lot in one
    # constraint but not others) should be slightly deprioritized unless they have high prize
    # Compute per-constraint consumption ratio
    # For each item, how much of the capacity does it consume in the tightest constraint?
    max_consumption = np.zeros(n_items, dtype=np.float64)
    for c in range(n_constraints):
        # Fraction of capacity consumed in constraint c
        frac = weights[c] / capacity
        max_consumption = np.maximum(max_consumption, frac)

    # Items consuming > 40% of any single capacity are risky; dampen slightly
    # unless their prize is very high
    risk_mask = max_consumption > 0.4
    if np.any(risk_mask):
        # Only dampen if prize isn't in top 20%
        prize_threshold = np.percentile(total_p, 80)
        dampen_mask = risk_mask & (total_p < prize_threshold)
        scores[dampen_mask] *= 0.85

    # Ensure all scores are positive and finite
    scores = np.where(np.isfinite(scores) & (scores > 0), scores, 1e-9)

    return scores
# EVOLVE-BLOCK-END
