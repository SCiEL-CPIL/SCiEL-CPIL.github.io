"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 300s limit, leave 10s margin
    
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes)
    weights = np.asarray(weights)
    
    # Reshape if 1D, though contract says (5, 100)
    if prizes.ndim == 1:
        prizes = prizes.reshape(1, -1)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
        
    # prizes: (5, 100), weights: (5, 100)
    # 1. Calculate total prize per item
    prize_sum = np.sum(prizes, axis=0) # shape (100,)
    
    # 2. Calculate the bottleneck weight per item (max weight across constraints)
    # This represents the minimum capacity required in the tightest constraint.
    max_weight = np.max(weights, axis=0) # shape (100,)
    
    # 3. Compute raw efficiency score
    # Avoid division by zero
    eps = 1e-9
    raw_scores = prize_sum / (max_weight + eps)
    
    # Handle non-finite values in raw scores
    raw_scores = np.where(np.isfinite(raw_scores), raw_scores, 0.0)
    
    # 4. Structural Change: Rank-Based Heuristic
    # Convert cardinal efficiency scores to ordinal ranks.
    # This smooths the heuristic landscape, preventing extreme outliers
    # from dominating the ant selection probability too aggressively.
    
    # argsort returns indices that would sort the array.
    # We want the best (highest score) to have rank 1.
    # np.argsort sorts in ascending order by default.
    # So, if we sort ascending, the last element is the best.
    
    # To get rank 1 for the best:
    # 1. Sort indices by score descending.
    # 2. Assign rank 1, 2, ... to these indices.
    
    # Efficient way to get ranks (1-based, 1 is best)
    # argsort of argsort gives the rank in the sorted order (0-based ascending).
    # If we want 1-based descending rank:
    
    # Step 1: Get order of indices if sorted by score descending
    # np.argsort(-raw_scores) gives indices of elements in descending order.
    # Let's call this 'order'.
    # The item at order[0] is the best. It should get rank 1.
    # The item at order[1] is the second best. It should get rank 2.
    
    # Create an array of ranks 1..N
    ranks = np.arange(1, len(raw_scores) + 1)
    
    # Map ranks to the original indices
    # heuristic_scores[order[i]] = ranks[i]
    # But we want the heuristic to be INVERSE rank (higher is better).
    # So, best item (rank 1) gets high heuristic.
    # Let's use 1/rank.
    
    heuristic_scores = np.zeros_like(raw_scores)
    order = np.argsort(-raw_scores)
    
    # Assign inverse ranks
    # For i in range(len(order)):
    #    idx = order[i]
    #    heuristic_scores[idx] = 1.0 / (i + 1)
    
    # Vectorized version:
    heuristic_scores[order] = 1.0 / ranks
    
    # 5. Safety checks
    # Ensure all values are positive and finite
    heuristic_scores = np.where(np.isfinite(heuristic_scores) & (heuristic_scores > 0), heuristic_scores, 1e-9)
    
    # Check deadline
    if time.time() > deadline:
        pass # Return whatever we have
        
    return heuristic_scores
# EVOLVE-BLOCK-END
