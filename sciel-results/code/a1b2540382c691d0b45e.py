# EVOLVE-BLOCK-START
import numpy as np

def priority(adjacency):
    """
    Computes a priority score for each vertex based on local graph structure
    (Degree and Clustering Coefficient).
    
    Approach:
    1. Convert adjacency to a boolean numpy array.
    2. Compute the degree of each vertex.
    3. Compute the local clustering coefficient for each vertex.
       The clustering coefficient measures the extent to which neighbors of a vertex
       are connected to each other.
    4. Score vertices such that those with high clustering (dense local neighborhood)
       and low degree (fewer neighbors to block) are prioritized.
       Rationale: Picking a vertex in a dense cluster is "efficient" because its
       neighbors are already conflicting with each other, so blocking them doesn't
       reduce the potential MIS size as much as blocking independent neighbors.
       Conversely, picking a vertex with many independent neighbors (low clustering)
       blocks many potential MIS candidates.
    5. Score formula: (1 + Clustering) / (1 + Degree)
    """
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency)
        
    # Ensure it's a boolean matrix for logical operations
    adj_bool = adjacency.astype(bool)
    N = adj_bool.shape[0]
    
    # 1. Compute Degrees
    degrees = adj_bool.sum(axis=1).astype(float)
    
    # 2. Compute Local Clustering Coefficients
    # C_v = 2 * E(N_v) / (d_v * (d_v - 1))
    # If d_v < 2, C_v = 0 (by convention, or undefined, but 0 is safe for scoring)
    
    clustering = np.zeros(N)
    
    # Pre-compute neighbor lists for efficiency
    neighbors = [np.where(adj_bool[i])[0] for i in range(N)]
    
    for v in range(N):
        d = int(degrees[v])
        if d < 2:
            clustering[v] = 0.0
            continue
        
        nbs = neighbors[v]
        # Count edges between neighbors
        # For each pair of neighbors (u, w) with u < w, check if adj[u, w] is True
        # We can use the submatrix adj_bool[np.ix_(nbs, nbs)]
        sub_adj = adj_bool[np.ix_(nbs, nbs)]
        # The number of edges is half the sum of the upper triangle (excluding diagonal)
        # Or simply sum of the matrix divided by 2 (since it's symmetric and diagonal is 0)
        edge_count = np.sum(sub_adj) / 2.0
        
        # Max possible edges is d * (d - 1) / 2
        max_edges = d * (d - 1) / 2.0
        
        if max_edges > 0:
            clustering[v] = edge_count / max_edges
        else:
            clustering[v] = 0.0
            
    # 3. Calculate Scores
    # We want to pick vertices that are "good" for MIS.
    # Good = Low Degree (less blocking) AND High Clustering (neighbors block each other anyway).
    # Score = (1 + Clustering) / (1 + Degree)
    
    # Add a small epsilon to degree to avoid division by zero (though degree is int >= 0)
    # If degree is 0, score = 1/1 * (1+0) = 1.
    # If degree is 1, score = 1/2 * (1+0) = 0.5.
    
    scores = (1.0 + clustering) / (1.0 + degrees)
    
    return scores
# EVOLVE-BLOCK-END
