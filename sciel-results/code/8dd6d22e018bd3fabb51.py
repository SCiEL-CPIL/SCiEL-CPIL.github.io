"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute one attractiveness score per item for ACO of multi-constraint knapsack.
    
    prizes: (5, 100) or (100,) uniform [0,1]
    weights: (5, 100) uniform [0,1]
    Returns: (100,) array, larger = more attractive.
    """
    # Deadline guard: if this function is called repeatedly, we need to be fast.
    # But the contract says "The heuristic is computed once per instance, before the colony runs."
    # So it's called 3 times total (once per instance). 
    # However, to be safe against any weird evaluation loop, we add a guard.
    # But since it's just math, it's already very fast.
    
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
    
    # Alternative heuristic idea: 
    # Instead of max weight (which is the worst case), maybe use sum of weights?
    # Or maybe use a different power?
    # Let's try to slightly adjust the heuristic to favor items that are good in ALL constraints,
    # not just the max one. 
    # Actually, the max weight is the bottleneck for fitting. 
    # But the "cost" of an item in MCKP is often measured by how much it consumes capacity.
    # Using max_w is a reasonable proxy for "how much capacity it eats up in the tightest dimension".
    
    # Let's try a small improvement:
    # The current heuristic is p / max_w.
    # If we use p / (sum_w / 5) or something? No, max_w is better for "fits" logic.
    
    # What if we scale it slightly? 
    # The pheromone starts at 1. If h is very large, it dominates.
    # Let's try to normalize h so that the mean is 1.
    # This makes the initial pheromone and heuristic balanced.
    mean_h = np.mean(h)
    if mean_h > 0:
        h = h / mean_h
    
    return h
# EVOLVE-BLOCK-END
