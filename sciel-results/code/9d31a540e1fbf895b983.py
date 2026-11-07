"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # prizes shape: (5, 100), weights shape: (5, 100)
    # Each item has a prize vector (5,) and weight vector (5,)
    # Capacity is 25 for each constraint
    
    # Sum prizes across constraints to get total prize per item
    total_prizes = prizes.sum(axis=0)  # shape: (100,)
    
    # For each item, compute the "effective weight" considering all constraints
    # An item that uses a lot of any single constraint is more costly
    # Use the max weight across constraints as a measure of bottleneck
    max_weight = weights.max(axis=0)  # shape: (100,)
    
    # Also consider total weight across constraints
    total_weight = weights.sum(axis=0)  # shape: (100,)
    
    # Combine: prefer items with high total prize and low bottleneck weight
    # Use a combined cost that penalizes both high max weight and high total weight
    cost = 0.5 * max_weight + 0.5 * total_weight
    
    # Heuristic: prize density with respect to combined cost
    h = total_prizes / (cost + 1e-9)
    
    # Ensure positivity
    h[h <= 0] = 1e-9
    
    return h
# EVOLVE-BLOCK-END
