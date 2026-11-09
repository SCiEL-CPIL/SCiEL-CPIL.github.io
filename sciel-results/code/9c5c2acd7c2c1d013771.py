import numpy as np
import time

def priority(adjacency):
    """
    Calculates a priority score for each vertex to maximize the size of the
    greedy independent set.
    
    Heuristic:
    1. Primary: Prefer vertices with lower degree (fewer neighbors to block).
    2. Secondary: Prefer vertices whose neighbors are densely connected (cliquish).
       Rationale: If neighbors are densely connected, they cannot be in the independent
       set together. Blocking them is "cheap" because they would have blocked each
       other anyway. If neighbors are sparsely connected, they are valuable candidates
       for the independent set, so blocking them is "expensive".
       
    Score = -degree(v) + induced_edges(N(v))
    """
    start_time = time.time()
    
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    
    # Calculate degrees
    degrees = adj.sum(axis=1).astype(np.float64)
    
    # Calculate induced edges in neighborhood for each vertex
    # E[v] = sum_{u in N(v)} sum_{w in N(v)} A[u, w]
    # We divide by 2 at the end or just use the sum as a proxy (monotonic)
    induced_edges = np.zeros(n, dtype=np.float64)
    
    for v in range(n):
        # Get neighbors of v
        # A[:, v] is the vector indicating neighbors
        # We want the sum of A[u, w] for all u, w that are neighbors of v
        # This is equivalent to:
        # mask = (adj[:, v] == 1)
        # sub_adj = adj[mask][:, mask]
        # induced_edges[v] = sub_adj.sum()
        
        # Optimized vectorized approach for small N:
        # Let col = adj[:, v]
        # We want col.T @ adj @ col
        col = adj[:, v]
        # adj @ col gives for each u, the number of neighbors of u that are neighbors of v
        # Then col.T @ (adj @ col) sums this over u in N(v)
        # This gives sum_{u in N(v)} sum_{w in N(v)} A[u, w]
        induced_edges[v] = col @ adj @ col
        
    # Normalize induced edges?
    # The term col @ adj @ col is 2 * E(N(v)).
    # We want to maximize this.
    
    # Define priority
    # We want to minimize degree and maximize induced edges.
    # Weights:
    # Degree range [0, 199]
    # Induced edges range [0, ~200^2]
    # For ER(200, 0.1), deg~20, induced~20.
    # So 1:1 weighting is reasonable.
    
    priorities = -degrees + induced_edges
    
    # Deadline guard
    if time.time() - start_time > 110:
        return priorities.astype(float)
        
    return priorities.astype(float)
