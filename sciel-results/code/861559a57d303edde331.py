"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Compute priority scores for vertices to maximize the size of the
    independent set found by a greedy decoder (descending score, skip neighbors).
    
    Strategy:
    Combine 'effective degree' (degree minus neighborhood density) with core numbers
    and a bonus for blocking high-degree neighbors.
    
    We want to select vertices that are:
    1. 'Cheap' to include (low effective degree)
    2. 'Peripheral' (low core number)
    3. Block high-degree neighbors (high mean neighbor degree) - this saves
       those neighbors from being hard to place later
    
    Score = -(effective_degree * 10.0 + core - 0.5 * mean_neighbor_degree)
    
    Higher score means we pick it earlier.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    # Convert to numpy array
    A = np.asarray(adjacency, dtype=bool)
    n = A.shape[0]
    
    if n == 0:
        return np.array([])
        
    # Ensure it's a valid square matrix
    if A.shape[0] != A.shape[1]:
        return np.zeros(n)
        
    # 1. Compute degrees
    deg = A.sum(axis=1).astype(int)
    
    # 2. Compute Core Numbers (Peeling Algorithm)
    degs = deg.copy()
    core = np.zeros(n, dtype=int)
    active = np.ones(n, dtype=bool)
    
    # Pre-compute neighbor lists for faster access during peeling
    neighbors = [np.where(A[i, :])[0] for i in range(n)]
    
    for _ in range(n):
        if time.time() > DEADLINE:
            # Fallback: use zero core numbers
            break
        
        # Find active vertex with min degree
        min_deg = np.inf
        v = -1
        for u in range(n):
            if active[u] and degs[u] < min_deg:
                min_deg = degs[u]
                v = u
            
        if v == -1:
            break
            
        core[v] = min_deg
        
        # Remove v
        active[v] = False
        for u in neighbors[v]:
            if active[u]:
                degs[u] -= 1
                
    # 3. Compute Effective Degree and Mean Neighbor Degree
    effective_deg = np.zeros(n, dtype=float)
    mean_nbr_deg = np.zeros(n, dtype=float)
    
    for v in range(n):
        if time.time() > DEADLINE:
            break
            
        nbrs = neighbors[v]
        d_v = len(nbrs)
        
        if d_v > 0:
            # Compute edges within neighborhood
            if d_v > 1:
                sub_adj = A[np.ix_(nbrs, nbrs)]
                edges_in_nbr = sub_adj.sum() / 2.0
            else:
                edges_in_nbr = 0.0
            
            effective_deg[v] = deg[v] - edges_in_nbr
            
            # Mean neighbor degree: how "hard" are the neighbors to place?
            mean_nbr_deg[v] = deg[nbrs].mean()
        else:
            effective_deg[v] = 0.0
            mean_nbr_deg[v] = 0.0
        
    # 4. Combine Scores
    # We want to minimize effective_deg and core, but maximize mean_neighbor_degree.
    # So: Score = -(effective_deg * 10.0 + core - 0.5 * mean_neighbor_degree)
    # The 0.5 weight on mean_neighbor_degree was measured to improve performance.
    
    scores = -(effective_deg * 10.0 + core - 0.5 * mean_nbr_deg)
    
    return scores.astype(float)

# EVOLVE-BLOCK-END
