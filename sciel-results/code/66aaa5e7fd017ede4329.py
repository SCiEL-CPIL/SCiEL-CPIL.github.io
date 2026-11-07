"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes a heuristic edge-attractiveness matrix for ACO-CVRP.
    
    Strategy:
    Use a softened inverse distance heuristic (1 / sqrt(d)).
    This reduces the "greediness" of the heuristic compared to 1/d or 1/d^2,
    allowing the pheromone trail to play a larger role in guiding the ants.
    This is particularly useful in short ACO runs where pheromone convergence
    is the primary driver of solution quality improvement.
    """
    d_mat = np.asarray(distance_matrix, dtype=np.float64)
    n = d_mat.shape[0]
    
    epsilon = 1e-9
    
    # 1. Softened Inverse Distance: 1 / sqrt(d)
    # This is less aggressive than 1/d, promoting exploration over pure greedy nearest-neighbor selection.
    base = 1.0 / np.sqrt(d_mat + epsilon)
    
    # Set diagonal to 0 to avoid self-loops
    np.fill_diagonal(base, 0.0)
    
    # 2. Safety: Ensure all values are positive and finite
    # Replace any non-finite or non-positive values with a small epsilon
    base = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
    
    # 3. Normalization
    # Normalize so that the mean of the positive entries is 1.0.
    # This ensures the heuristic magnitude is consistent across instances,
    # preventing numerical issues if the coordinate scale varies slightly.
    positive_vals = base[base > 0]
    if len(positive_vals) > 0:
        mean_val = np.mean(positive_vals)
        if mean_val > 0:
            base = base / mean_val
    
    # Final safety check
    base = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
    
    return base
# EVOLVE-BLOCK-END
