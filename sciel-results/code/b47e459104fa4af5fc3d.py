"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Computes attractiveness scores for items in a Multiple-Constraint Knapsack Problem.
    
    Strategy: Multi-Criteria Resource Scarcity Ranking.
    1. Normalize each constraint's weight by its capacity (25) to get relative consumption.
    2. For each item, find the maximum relative consumption (the tightest resource it uses).
    3. Score items by prize / (max_relative_consumption + epsilon).
    4. Normalize scores to a stable range to avoid pheromone explosion/collapse.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin for 300s limit

    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if time.time() > deadline:
        return np.ones(prizes.shape[1])

    # Capacity is fixed at 25 for all constraints
    capacity = 25.0
    
    # Normalize weights to [0, 1] relative to capacity
    # This measures how much of the total available resource an item consumes
    normalized_weights = weights / capacity
    
    # For each item, find the maximum normalized weight across all constraints
    # This represents the "bottleneck" resource in relative terms
    max_relative_weights = np.max(normalized_weights, axis=0)
    
    # Calculate heuristic: prize / bottleneck_relative_weight
    # Items that consume a large fraction of a tight resource are penalized more heavily
    epsilon = 1e-9
    raw_scores = prizes / (max_relative_weights + epsilon)
    
    # Handle non-finite values
    raw_scores = np.where(np.isfinite(raw_scores), raw_scores, 0.0)
    
    # Ensure positivity
    raw_scores = np.where(raw_scores > 0, raw_scores, epsilon)
    
    # Normalize scores to a reasonable range [1, 10] to stabilize ACO
    # This prevents extreme ratios from dominating the pheromone update
    min_score = np.min(raw_scores)
    max_score = np.max(raw_scores)
    
    if max_score > min_score:
        # Linear normalization to [1, 10]
        normalized_scores = 1.0 + 9.0 * (raw_scores - min_score) / (max_score - min_score)
    else:
        normalized_scores = np.ones_like(raw_scores)
        
    # Final safety check for finite values
    normalized_scores = np.where(np.isfinite(normalized_scores), normalized_scores, 1.0)
    normalized_scores = np.where(normalized_scores > 0, normalized_scores, 1e-9)
    
    return normalized_scores
# EVOLVE-BLOCK-END
