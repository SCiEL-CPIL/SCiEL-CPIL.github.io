"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    start_time = time.time()
    deadline = 300.0 - 5.0  # Safety margin

    # Convert to numpy arrays for consistent behavior
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Handle potential 2D prizes by reducing to 1D if necessary
    if prizes.ndim == 2:
        prize_vector = np.sum(prizes, axis=0)
    else:
        prize_vector = prizes.flatten()

    # Calculate sum and max weights per item
    weight_sums = np.sum(weights, axis=0)
    weight_maxs = np.max(weights, axis=0)
    
    # Smooth bottleneck penalty: use a soft-max-like approach.
    # For items where one constraint dominates (max/sum is high), add extra penalty.
    # ratio = max/sum; when ratio is close to 1, the item is bottleneck-heavy.
    # Use (ratio - 0.2)^2 * weight_sums as a smooth penalty that only kicks in
    # for items where a single constraint uses >20% of total weight.
    ratio = weight_maxs / (weight_sums + 1e-9)
    bottleneck_penalty = np.maximum(ratio - 0.2, 0.0) ** 2 * weight_sums
    
    # Cost: total weight + smooth bottleneck penalty
    cost = weight_sums + bottleneck_penalty + 1e-9
    
    # Base heuristic: prize / cost
    heuristic = prize_vector / cost
    
    # Apply power amplification: 1.8 for sharper ACO focus
    heuristic = heuristic ** 1.8
    
    # Ensure positive and finite
    heuristic = np.maximum(heuristic, 1e-9)
    heuristic = np.nan_to_num(heuristic, nan=1e-9, posinf=1e9, neginf=1e-9)
    
    # Check time
    if time.time() > deadline:
        return heuristic

    return heuristic

# EVOLVE-BLOCK-END
