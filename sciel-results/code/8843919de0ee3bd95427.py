"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute one attractiveness score per item for ACO of multi-constraint knapsack.
    
    prizes: (5, 100) or (100,) uniform [0,1]
    weights: (5, 100) uniform [0,1]
    Returns: (100,) array, larger = more attractive.
    """
    # Ensure prizes is 1D of length 100
    if prizes.ndim == 2:
        # Take mean across constraints for a single prize per item
        p = np.mean(prizes, axis=0)
    else:
        p = np.asarray(prizes, dtype=float).ravel()
    
    weights = np.asarray(weights, dtype=float)
    
    # Calculate the maximum weight per item across all constraints (bottleneck)
    max_w = np.max(weights, axis=0)
    
    # Heuristic: Prize / Bottleneck Weight
    # Items with high prize and low bottleneck weight are more attractive.
    # This identifies items that are "cheap" in their most constraining dimension.
    h = p / (max_w + 1e-9)
    
    # Ensure positive values as per contract: "Heuristic entries that are not positive become 1e-9"
    h = np.where(h > 0, h, 1e-9)
    
    # Normalize to ensure reasonable magnitude? 
    # The contract says pheromone starts at 1 and heuristic is multiplied.
    # If heuristic is very large, it dominates. If very small, pheromone dominates.
    # Typical ACO heuristics are often normalized or just used as-is.
    # The seed score 0.9943 suggests the current scale is close to optimal.
    # Let's keep the raw ratio but maybe add a small normalization or scaling factor if needed.
    # However, changing the scale changes the relative weight of pheromone vs heuristic.
    # Since pheromone starts at 1, if heuristic is ~1 (mean prize 0.5 / mean max_w 0.5 = 1), it's balanced.
    # Uniform [0,1] mean is 0.5. So h ~ 1 on average. This is a good scale.
    
    return h
# EVOLVE-BLOCK-END
