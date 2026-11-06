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
    
    # Strategy: Balance degree and mean neighbor degree
    # Lower degree is better.
    # Higher mean neighbor degree is better (blocking expensive neighbors).
    # Previous weight of 10.0 on degree made MND negligible.
    # Try equal weighting to let structural context influence the choice.
    
    scores = -deg + mean_neighbor_deg
    
    return scores
