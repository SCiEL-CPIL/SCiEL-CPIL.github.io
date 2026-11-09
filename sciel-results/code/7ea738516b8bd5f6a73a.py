import numpy as np
import time

def priority(adjacency):
    """
    Assigns scores to vertices to maximize the size of the independent set
    produced by a greedy decoder that picks vertices in descending score order.
    
    Heuristic:
    We prioritize vertices that have a low degree themselves (so they don't block
    many candidates) but have neighbors with high degrees (so picking them prevents
    those high-degree neighbors from being picked, which would block many other vertices).
    
    Score(v) = -degree(v) + sum(degree(u) for u in neighbors(v))
    """
    start_time = time.time()
    deadline = start_time + 10.0
    
    # Convert to float for matrix operations
    adj = np.asarray(adjacency, dtype=float)
    
    # Calculate degrees
    degrees = adj.sum(axis=1)
    
    # Calculate sum of degrees of neighbors for each vertex
    # This is equivalent to adj @ degrees
    neighbor_degree_sums = adj.dot(degrees)
    
    # Compute scores
    # High score means "pick me first"
    # We want to pick vertices with low degree (low blocking cost)
    # and high neighbor degree sum (high benefit of blocking those neighbors)
    scores = -degrees + neighbor_degree_sums
    
    # Safety check
    if time.time() > deadline:
        # Fallback to simple negative degree
        return -degrees
        
    return scores
