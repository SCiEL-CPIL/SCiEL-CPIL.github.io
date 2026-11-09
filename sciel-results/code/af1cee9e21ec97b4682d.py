"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # prizes shape (5, 100), weights shape (5, 100)
    # Each constraint has capacity 25.
    # We want to score items by their "value density" considering all constraints.
    # An item is good if it has high prize and low weight across constraints.
    # The bottleneck constraint is the one that limits the item the most.
    # Using max weight per item captures the tightest constraint.
    
    # Total prize per item (sum across constraints? No, prizes are per constraint? 
    # Wait, "prizes and weights, of shape (5, 100), are uniform in [0, 1]."
    # Usually in knapsack, prize is a single value per item. But here shape is (5, 100).
    # The task says "highest total prize found by any ant".
    # In ACO, when an ant picks an item, it adds "prize".
    # If prizes is (5,100), does the total prize sum over the 5 constraints?
    # Or is the prize for item j the sum of prizes[i, j]?
    # Let's assume the value of an item is the sum of its prize components, or maybe the max?
    # Given the previous code `prizes / (weights.sum(axis=0) + 1e-9)`, it treated prizes as a matrix and divided by scalar.
    # This would result in a matrix of heuristics?
    # "returns one attractiveness score per item" -> shape must be (100,).
    # The previous code returned `prizes / (max_weight + 1e-9)`.
    # If `prizes` is (5,100) and `max_weight` is (100,), broadcasting makes it (5,100).
    # This would be WRONG shape for "one score per item".
    # Let's re-read: "heuristics(prizes, weights), which returns one attractiveness score per item".
    # So the return shape MUST be (100,).
    
    # How to aggregate prizes?
    # If the "total prize" of a solution is the sum of prizes of selected items,
    # and each item has a prize vector of size 5, then the total prize of a solution
    # is likely the sum of the item prize vectors, and then maybe summed or maxed?
    # "highest total prize found by any ant"
    # Usually, total prize is a scalar.
    # If prizes are (5,100), maybe the prize of item j is sum(prizes[:, j])?
    # Or maybe the prize is just one row? No, "prizes and weights, of shape (5, 100)".
    # Let's assume the value of an item is the sum of its prizes across the 5 dimensions.
    # And the cost is the weights across 5 dimensions.
    # The heuristic should reflect value/cost ratio.
    
    item_prizes = prizes.sum(axis=0)  # Shape (100,)
    # For cost, using max weight is a good approximation of the bottleneck.
    # Or sum of weights?
    # If an item has weight [0.9, 0.1, 0.1, 0.1, 0.1], it's limited by the first constraint.
    # If it has [0.5, 0.5, 0.5, 0.5, 0.5], it's balanced.
    # Max weight is a safe upper bound on the "usage" of any single constraint.
    # Let's try summing prizes and dividing by max weight.
    
    max_weight = weights.max(axis=0)
    # Avoid division by zero
    heuristic = item_prizes / (max_weight + 1e-9)
    
    # Ensure positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
