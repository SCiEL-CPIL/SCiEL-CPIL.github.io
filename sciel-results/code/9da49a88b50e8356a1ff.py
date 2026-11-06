"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    # Start time tracking for safety, though computation is fast
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s/150s limits

    # Convert to float for matrix operations if needed, but sum is integer
    # adjacency is a bool matrix
    n = adjacency.shape[0]
    
    # Calculate degrees
    deg = adjacency.sum(axis=1).astype(float)
    
    # Calculate sum of neighbor degrees
    # neighbor_deg_sum[i] = sum(deg[j] for j in neighbors of i)
    neighbor_deg_sum = adjacency @ deg
    
    # Calculate mean neighbor degree
    # Avoid division by zero
    mean_neighbor_deg = np.zeros(n)
    mask = deg > 0
    mean_neighbor_deg[mask] = neighbor_deg_sum[mask] / deg[mask]
    
    # Heuristic:
    # We want low degree (primary)
    # And high mean neighbor degree (secondary, to remove 'bad' high-degree neighbors)
    # The knowledge suggests this improves score.
    # Tuning alpha: p8 used 0.1 and failed (score 0.94). 
    # Perhaps the effect was too weak or the specific implementation was flawed.
    # Let's try a stronger weight. 
    # If deg is ~10 and mnd is ~10, -10 + 0.1*10 = -9.
    # If deg is 11 and mnd is 10, -11 + 0.1*10 = -10.
    # Difference is 1.0. 
    # If we use alpha=1.0:
    # -10 + 10 = 0.
    # -11 + 10 = -1.
    # Difference is 1.0.
    # The relative ranking might be more sensitive with higher alpha.
    
    # Let's try alpha = 0.5 as a middle ground, or even 1.0.
    # Given p8 (0.1) was worse than seed (0.0 effectively), maybe the "reward" for high mnd was not enough to overcome the penalty of slightly higher degree, OR the penalty was too strong relative to the reward.
    # Actually, if p8 was worse, it means the heuristic *hurt* performance.
    # Why? Maybe picking nodes with high MND is bad?
    # Let's re-read the logic.
    # "preferring those whose exclusion of high-degree neighbors is less costly"
    # If I exclude high-degree neighbors, is that costly?
    # High-degree neighbors are "expensive" to leave in the graph because they block many others.
    # So removing them is GOOD.
    # So High MND should be GOOD.
    # Why did p8 fail?
    # Maybe the alpha was too low to matter, and the noise/implementation error caused it?
    # Or maybe the specific graphs have a structure where MND is negatively correlated with "goodness"?
    
    # Let's look at the "1 -> 1.008" claim. It says "Measured".
    # This implies a *different* implementation or tuning worked.
    # The p8 code in the JSON is just one attempt.
    # I will try alpha = 0.2.
    
    alpha = 0.2
    scores = -deg + alpha * mean_neighbor_deg
    
    # Check deadline (should never trigger for N=200)
    if time.time() > deadline:
        return -deg.astype(float) # Fallback to seed
    
    return scores
# EVOLVE-BLOCK-END
