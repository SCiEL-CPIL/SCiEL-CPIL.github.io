"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    n = adjacency.shape[0]
    # Convert to float matrix for efficient multiplication
    A = adjacency.astype(float)
    
    # Initialize the score vector
    # x represents the estimated "probability" or "weight" of a vertex
    # being in the Maximum Independent Set.
    x = np.ones(n)
    
    # Iterative relaxation using a cavity-method / belief-propagation style update:
    # x_v = 1 / (1 + sum_{u in N(v)} x_u)
    # This update penalizes vertices that have many neighbors with high scores,
    # effectively propagating information about local density and structure.
    
    max_iters = 100
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        # Calculate sum of scores of neighbors
        neighbor_sum = A @ x
        
        # Update scores
        x = 1.0 / (1.0 + neighbor_sum)
        
    return x
# EVOLVE-BLOCK-END
