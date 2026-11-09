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

    # Ensure weights is (5, 100)
    # Calculate sum and max weights per item
    weight_sums = np.sum(weights, axis=0)
    weight_maxs = np.max(weights, axis=0)
    
    # Refined cost function:
    # Stronger penalty for bottleneck-heavy items (max) on top of total bulk (sum).
    # 0.8 coefficient increases the relative penalty for spiky weight distributions,
    # which are riskier in multi-constraint problems with equal capacities.
    cost = weight_sums + 0.8 * weight_maxs + 1e-9
    
    # Base heuristic
    heuristic = prize_vector / cost
    
    # Apply a sharper power to focus the ACO on top candidates more aggressively.
    # 1.5 power creates a steeper probability distribution, aiding rapid convergence
    # to high-value items in early iterations, which reinforces pheromone trails.
    heuristic = heuristic ** 1.5
    
    # Ensure positive and finite
    heuristic = np.maximum(heuristic, 1e-9)
    heuristic = np.nan_to_num(heuristic, nan=1e-9, posinf=1e9, neginf=1e-9)
    
    # Check time
    if time.time() > deadline:
        return heuristic

    return heuristic

# EVOLVE-BLOCK-END
