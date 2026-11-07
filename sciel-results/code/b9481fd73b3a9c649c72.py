import numpy as np

def heuristics(prizes, weights):
    # Aggregate prizes to get a single score per item
    total_prize = prizes.sum(axis=0)
    
    # Calculate the maximum weight in any single constraint (L-infinity norm)
    # This structural change focuses on the bottleneck constraint for each item.
    # Items with a very high weight in one constraint are heavily penalized,
    # which is critical in multiple-constraint knapsack problems where a single
    # tight constraint can render an item unusable.
    weight_max = np.max(weights, axis=0)
    
    # Return heuristic scores
    return total_prize / (weight_max + 1e-9)
