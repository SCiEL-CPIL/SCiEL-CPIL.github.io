"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Score vertices for greedy Maximum Independent Set.
    
    Uses a lexicographic scoring system based on structural properties:
    1. Min Core Number (Primary): Vertices in sparser cores (periphery) are safer to pick.
    2. Min Degree (Secondary): Lower degree means fewer conflicts.
    3. Max Mean Neighbor Degree (Tertiary): Picking a vertex whose neighbors are high-degree
       is less costly, as those neighbors are likely to be excluded anyway.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1).astype(float)
    
    # Compute Core Numbers using iterative peeling (O(N^2), fast for N=200)
    deg_int = adjacency.sum(axis=1).astype(int).copy()
    core_nums = np.zeros(n, dtype=int)
    active = np.ones(n, dtype=bool)
    
    for _ in range(n):
        min_deg = np.inf
        min_idx = -1
        # Find active vertex with minimum degree
        for i in range(n):
            if active[i] and deg_int[i] < min_deg:
                min_deg = deg_int[i]
                min_idx = i
        
        if min_idx == -1:
            break
            
        core_nums[min_idx] = min_deg
        active[min_idx] = False
        
        # Update degrees of active neighbors
        neighbors = np.where(adjacency[min_idx] > 0)[0]
        for v in neighbors:
            if active[v]:
                deg_int[v] -= 1

    # Mean degree of neighbors
    neighbor_degree_sums = degrees @ adjacency
    mean_neighbor_degrees = neighbor_degree_sums / (degrees + 1)
    
    # Lexicographic score:
    # 1. Min Core Number (Weight 10000)
    # 2. Min Degree (Weight 1000)
    # 3. Max Mean Neighbor Degree (Weight 1)
    score = -10000.0 * core_nums - 1000.0 * degrees + mean_neighbor_degrees
    
    # Time guard
    if time.time() - start_time > deadline:
        # Fallback to simple degree-based score if core calculation was slow (unlikely)
        return -degrees
    
    return score
# EVOLVE-BLOCK-END
