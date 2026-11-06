import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n = adjacency.shape[0]
    deg = adjacency.sum(axis=1).astype(float)
    
    # Avoid division by zero
    safe_deg = deg + 1.0
    
    # Calculate sum of 1/deg for neighbors
    inv_deg = 1.0 / safe_deg
    neighbor_sum_inv_deg = adjacency @ inv_deg
    
    # Calculate mean neighbor degree
    neighbor_deg_sum = adjacency @ deg
    mean_neighbor_deg = np.zeros(n)
    mask = deg > 0
    mean_neighbor_deg[mask] = neighbor_deg_sum[mask] / deg[mask]
    
    # Strategy: Strongly prefer low degree, use MND as secondary signal
    # Scale degree to dominate, MND as tiebreaker
    # For ER(200, p), typical deg is around 10-30
    # We want to strongly penalize high degree vertices
    
    # Primary: minimize degree (strong penalty)
    # Secondary: minimize mean neighbor degree (prefer vertices whose neighbors have high degree)
    # This way, when we pick a low-degree vertex, its neighbors (which we can't pick) 
    # are likely to be high-degree, which is good because they'd have been "expensive" to include
    
    scores = -10.0 * deg + mean_neighbor_deg
    
    return scores
