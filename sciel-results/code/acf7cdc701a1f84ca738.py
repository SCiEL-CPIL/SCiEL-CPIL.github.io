"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    # Convert to numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Safety: ensure weights are positive to avoid division by zero
    weights_safe = np.maximum(weights, 1e-9)
    
    # 1. Calculate Base Heuristic: Total Prize * Bottleneck Slack
    # Total prize is the sum of prizes across the 5 constraints
    total_prize = np.sum(prizes, axis=0)
    
    # Bottleneck slack: min over constraints of (25 - weight) / weight
    # This penalizes items that are heavy in any single constraint
    slack = np.min((25.0 - weights_safe) / weights_safe, axis=0)
    
    # Base score
    base_heuristic = total_prize * slack
    base_heuristic = np.maximum(base_heuristic, 1e-9)
    
    # 2. Frequency-weighted Greedy Simulation
    # Run a few stochastic greedy simulations to boost items that are frequently selected
    # in feasible high-value solutions.
    
    n_items = weights.shape[1]
    counts = np.zeros(n_items, dtype=np.float64)
    
    # Number of greedy runs. 20 is fast and provides a decent frequency estimate.
    n_greedy_runs = 20
    
    for _ in range(n_greedy_runs):
        remaining_capacity = np.full(5, 25.0)
        selected = np.zeros(n_items, dtype=bool)
        
        while True:
            # Check which items still fit
            # weights is (5, 100), remaining_capacity is (5,)
            # Broadcasting: remaining_capacity[:, None] >= weights
            fits_mask = np.all(remaining_capacity[:, None] >= weights_safe, axis=0)
            
            # Exclude already selected items
            fits_mask &= ~selected
            
            if not np.any(fits_mask):
                break
            
            # Get indices of fitting items
            candidate_indices = np.where(fits_mask)[0]
            
            # Get base heuristic scores for candidates
            candidate_scores = base_heuristic[candidate_indices]
            
            # Normalize to probabilities
            total_score = np.sum(candidate_scores)
            if total_score <= 0:
                break
            probabilities = candidate_scores / total_score
            
            # Select one item
            chosen_idx = np.random.choice(candidate_indices, p=probabilities)
            
            # Update state
            selected[chosen_idx] = True
            remaining_capacity -= weights_safe[:, chosen_idx]
            counts[chosen_idx] += 1
            
    # 3. Combine Base Heuristic with Frequency
    # Items that are frequently picked get a boost.
    # Using (1 + counts) ensures all items have at least the base score.
    final_heuristic = base_heuristic * (1.0 + counts)
    
    # Ensure all values are positive
    final_heuristic = np.maximum(final_heuristic, 1e-9)
    
    return final_heuristic
# EVOLVE-BLOCK-END
