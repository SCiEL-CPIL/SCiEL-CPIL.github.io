"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute one attractiveness score per item for ACO of multi-constraint knapsack.
    
    prizes: (5, 100) uniform [0,1]
    weights: (5, 100) uniform [0,1]
    Returns: (100,) array, larger = more attractive.
    
    Strategy: total prize divided by the maximum weight across constraints.
    The max weight identifies the bottleneck constraint for each item.
    Items with high total prize and low worst-case weight score highest.
    """
    total_prize = np.sum(prizes, axis=0)       # (100,)
    max_weight = np.max(weights, axis=0)       # (100,)
    
    # Heuristic: prize-to-worst-weight ratio
    eta = total_prize / (max_weight + 1e-9)
    
    # Ensure positive finite values
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
