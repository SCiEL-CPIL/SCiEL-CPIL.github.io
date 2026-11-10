"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    # Deadline guard for the whole evaluation budget.
    start_time = time.time()
    deadline = 300.0 - 5.0

    # Convert inputs to float arrays.
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Reduce prizes to a 1-D vector of per-item total prize.
    if prizes.ndim == 2:
        prize_vector = np.sum(prizes, axis=0)
    else:
        prize_vector = prizes.flatten()

    # Per-item weight statistics across the 5 constraints.
    w_sum = np.sum(weights, axis=0)
    w_max = np.max(weights, axis=0)

    # Build a cost that penalises the bottleneck constraint more strongly.
    # The MCKP feasibility is dominated by the constraint where an item is
    # heaviest, so we weight the maximum above the total.
    # A small constant avoids division by zero and keeps values finite.
    cost = 0.6 * w_sum + 0.4 * w_max + 1e-12

    # Base heuristic: prize density.
    base = prize_vector / cost

    # Sharpen the distribution: a power > 1 biases the ACO towards the
    # highest value-density items early, which reinforces their pheromone
    # trails.  A power of 1.8 is a reasonable compromise between exploration
    # and exploitation for a 100-item, 5-constraint instance.
    heuristic = base ** 1.8

    # Guarantee positivity and finiteness.
    heuristic = np.maximum(heuristic, 1e-9)
    heuristic = np.nan_to_num(heuristic, nan=1e-9, posinf=1e9, neginf=1e-9)

    # If we are running out of wall-clock, return immediately.
    if time.time() > deadline:
        return heuristic

    return heuristic

# EVOLVE-BLOCK-END
