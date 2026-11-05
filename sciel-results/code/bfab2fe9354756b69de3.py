"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 120.0  # seconds
_SAFETY_MARGIN = 5.0  # seconds


def _is_deadline_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)


def priority(adjacency):
    """
    Compute priority scores based on local neighborhood properties.
    Heuristic: Prioritize vertices with low degree that are connected to high-degree
    neighbors (hubs). This allows the greedy decoder to 'sacrifice' high-degree
    vertices (which block many others) by picking their low-degree neighbors,
    thereby freeing up the rest of the graph.
    """
    n = adjacency.shape[0]
    adjacency = np.asarray(adjacency, dtype=bool)
    
    # Compute degrees
    degree = adjacency.sum(axis=1).astype(float)
    
    # Compute sum of neighbor degrees
    # A @ degree gives the sum of degrees of neighbors for each vertex
    neighbor_degree_sum = adjacency.astype(float) @ degree
    
    # Handle isolated nodes (degree 0)
    # For isolated nodes, neighbor_degree_sum is 0.
    # We want them to have high priority.
    
    # Calculate average neighbor degree, treating 0-degree nodes as having 0 avg neighbor degree
    # but we will add a bonus for low degree.
    avg_neighbor_degree = np.zeros(n)
    mask = degree > 0
    avg_neighbor_degree[mask] = neighbor_degree_sum[mask] / degree[mask]
    
    # Score:
    # 1. Low degree is good (less blocking)
    # 2. High average neighbor degree is good (blocks hubs)
    
    # Normalize features to [0, 1] for stable combination
    d_max = degree.max()
    if d_max > 0:
        degree_norm = degree / d_max
    else:
        degree_norm = np.zeros(n)
        
    an_max = avg_neighbor_degree.max()
    if an_max > 0:
        an_norm = avg_neighbor_degree / an_max
    else:
        an_norm = np.zeros(n)
        
    # Combined score
    # Weighting: The 'hub blocking' effect is strong.
    # Let's try: Score = -Degree + HubBlocking
    # If we pick a vertex with degree 1 connected to a hub (degree 50), we block the hub.
    # If we pick a vertex with degree 50, we block 50 vertices.
    # So picking the degree 1 vertex is much better.
    
    # Let's use: Score = avg_neighbor_degree - degree
    # This rewards low degree and high neighbor degree.
    
    score = an_norm - degree_norm
    
    # Add a small random tie-breaker to avoid symmetry issues in regular graphs?
    # No, deterministic is better for reproducibility, but ER graphs are random.
    # However, if two vertices have identical scores, the order matters.
    # We can add a tiny bit of noise or use index as tie-breaker.
    # The greedy decoder likely handles ties arbitrarily or by index.
    
    return score
# EVOLVE-BLOCK-END
