"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Degrees
    deg = A.sum(axis=1)
    
    # Sum of degrees of neighbors for each vertex
    neighbor_deg_sum = A @ deg
    
    # Mean degree of neighbors (avoid division by zero)
    safe_deg = np.maximum(deg, 1.0)
    mean_neighbor_deg = neighbor_deg_sum / safe_deg
    
    # Number of 2-step neighbors (neighbors of neighbors)
    # A2 = A @ A gives number of length-2 paths from v to u
    # For vertex v, the number of distinct 2-step neighbors is approximated by
    # (A @ A)[v, :] sum minus deg[v] (to exclude direct neighbors counted twice)
    # Actually, (A @ A)[v, v] = deg[v] (each neighbor contributes 1 to the diagonal)
    # (A @ A)[v, u] for u != v = number of common neighbors of v and u
    # Total 2-step neighbors from v: sum_u (A @ A)[v, u] - deg[v] (subtract direct neighbors)
    # But this counts each 2-step path, not distinct vertices. For ER graphs this is a good proxy.
    
    # Compute A @ A efficiently
    A2 = A @ A
    # For each vertex v, total number of 2-step paths starting from v
    two_step_paths = A2.sum(axis=1)
    # Subtract deg[v] because the diagonal of A2 is deg[v]
    # The off-diagonal sum gives paths to other vertices
    off_diag_two_step = two_step_paths - deg
    
    # "Clustering" of neighbors: if neighbors of v are densely connected, 
    # then picking v blocks fewer potential independent set members
    # This is related to the number of edges within N(v)
    # For vertex v, edges within N(v) = 0.5 * sum_{u in N(v)} (number of neighbors of u that are also in N(v))
    # = 0.5 * sum_{u in N(v)} (A2[v, u]) for u != v
    # Actually, A2[v, u] for u in N(v) counts common neighbors, which includes edges within N(v)
    # Let's use: for each v, sum of A2[v, u] over u in N(v) 
    # This can be computed as: for each v, sum over neighbors u of A2[v, u]
    # = row v of A2, masked by A[v, :]
    
    # For each vertex v, sum of A2[v, u] where A[v, u] = 1
    # This is (A * A2).sum(axis=1) element-wise: A[v,u] * A2[v,u] summed over u
    neighbor_clustering = (A * A2).sum(axis=1)
    
    # Normalize clustering by degree to get average "connectivity" of neighbors
    safe_deg2 = np.maximum(deg, 1.0)
    mean_neighbor_connectivity = neighbor_clustering / safe_deg2
    
    # Score components:
    # 1. -deg: prefer low degree (less blocking)
    # 2. + mean_neighbor_deg: prefer vertices whose neighbors have high degree 
    #    (blocking them saves more potential MIS members)
    # 3. - mean_neighbor_connectivity: prefer vertices whose neighbors are NOT densely connected
    #    (if neighbors are densely connected, they form a clique-like structure, 
    #     so picking one blocks many, but the others are mutually exclusive anyway)
    #    Actually, if neighbors are densely connected, they can't all be in MIS,
    #    so blocking them is less costly. So we should PREFER high neighbor connectivity.
    #    Let me reconsider: if N(v) is a clique, then at most 1 neighbor can be in MIS.
    #    So picking v blocks at most 1 potential MIS member. This is GOOD.
    #    So we want + mean_neighbor_connectivity
    
    # 4. - off_diag_two_step / deg: prefer vertices with fewer 2-step neighbors
    #    (fewer "distant" vertices that might be affected indirectly)
    mean_two_step = off_diag_two_step / safe_deg
    
    # Combined score
    score = -1.0 * deg + 0.8 * mean_neighbor_deg + 0.3 * mean_neighbor_connectivity - 0.2 * mean_two_step
    
    # Check deadline
    if time.time() > deadline:
        return score.astype(float)
    
    return score.astype(float)
# EVOLVE-BLOCK-END
