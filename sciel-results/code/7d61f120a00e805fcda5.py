"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    # Convert to numpy if needed
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])

    # Degree of each vertex
    degrees = adj.sum(axis=1).astype(int)
    
    # Compute Core Numbers
    # Algorithm: Process vertices in descending order of degree.
    # For each vertex v, its core number is its current degree.
    # Then, for each neighbor u, if deg[u] > core[v], set deg[u] = core[v].
    
    core = np.zeros(n, dtype=int)
    deg = degrees.copy()
    
    # Sort indices by degree descending
    sorted_indices = np.argsort(deg)[::-1]
    
    for v in sorted_indices:
        c = deg[v]
        core[v] = c
        # Update neighbors
        # Find neighbors of v
        neighbors = np.where(adj[v] > 0)[0]
        for u in neighbors:
            if deg[u] > c:
                deg[u] = c

    # Normalize core numbers and degrees to [0, 1]
    max_core = np.max(core)
    max_deg = np.max(degrees)
    
    if max_core == 0:
        max_core = 1
    if max_deg == 0:
        max_deg = 1
        
    norm_core = core / max_core
    norm_deg = degrees / max_deg
    
    # Heuristic:
    # Prefer vertices with LOW core number (peripheral) and LOW degree.
    # Core number captures the "centrality" in the densest subgraph.
    # Low core number vertices are safer to pick as they don't disrupt the core structure as much.
    
    score = - (norm_core + norm_deg)
    
    return score
# EVOLVE-BLOCK-END
