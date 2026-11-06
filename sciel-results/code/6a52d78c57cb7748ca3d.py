"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Assign a score to each vertex such that a greedy decoder picking vertices
    in descending score order (skipping neighbors of already-picked vertices)
    yields a large independent set.
    
    Strategy:
    1. Convert adjacency to numpy array if needed.
    2. Compute degree of each vertex.
    3. Use spectral relaxation: compute the principal eigenvector of the 
       adjacency matrix (or a related matrix) to capture global structure.
    4. Combine local degree information with global spectral information.
    
    Key insight: For an Erdos-Renyi graph, we want to select vertices that are
    'central' in a way that maximizes the independent set. The greedy algorithm
    picks high-score vertices first and skips their neighbors. So we want:
    - Vertices with fewer neighbors (lower degree) to have higher scores, because
      they block fewer other vertices.
    - But also consider global structure: vertices in sparse regions of the graph
      can form larger independent sets.
    
    Spectral approach: The principal eigenvector of the adjacency matrix gives
    a measure of 'centrality'. However, for independent sets, we might want to
    use the Laplacian or a modified adjacency matrix.
    
    Let's try: score = -degree + alpha * spectral_score
    where spectral_score is the principal eigenvector component.
    
    Actually, let me think more carefully. The greedy algorithm selects vertices
    in descending score order. If we pick a vertex, we skip all its neighbors.
    So a vertex with many neighbors is 'expensive' to pick because it blocks
    many others. A vertex with few neighbors is 'cheap'.
    
    But there's a trade-off: if all vertices have low degree, we can pick many.
    The spectral component might help identify which low-degree vertices are
    in regions where we can pick the most.
    
    Let's use: score = -degree * (1 + neighbor_degree_sum / max_degree) + beta * principal_eigenvector
    
    Or simpler: just use -degree combined with a spectral term.
    
    For the spectral term, I'll use power iteration to find the principal 
    eigenvector of the adjacency matrix (normalized).
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second limit
    
    # Convert to numpy array
    A = np.asarray(adjacency, dtype=np.float64)
    n = A.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Compute degree
    degree = A.sum(axis=1)
    
    # Compute neighbor degree sum (sum of degrees of neighbors)
    neighbor_degree_sum = A @ degree
    
    # Power iteration to find principal eigenvector of A
    # Initialize with random vector
    np.random.seed(42)
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    # Power iteration
    max_iter = 50
    for i in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        v_new = A @ v
        norm = np.linalg.norm(v_new)
        if norm < 1e-12:
            break
        v_new = v_new / norm
        # Check convergence
        if np.linalg.norm(v_new - v) < 1e-8:
            v = v_new
            break
        v = v_new
    
    # Now combine: we want low degree and high spectral centrality
    # But actually, for independent sets, we might want to de-prioritize
    # vertices that are in dense regions.
    
    # Let's try: score = -degree - neighbor_degree_sum / (n + 1) + alpha * v
    # where alpha is a scaling factor
    
    # Normalize v to have zero mean and unit variance for better combination
    v_mean = v.mean()
    v_std = v.std()
    if v_std > 1e-12:
        v_normalized = (v - v_mean) / v_std
    else:
        v_normalized = v - v_mean
    
    # Combine scores
    # Weight the terms
    deg_norm = degree / (n + 1)
    nds_norm = neighbor_degree_sum / (n * (n + 1))
    
    # score = -w1 * deg_norm - w2 * nds_norm + w3 * v_normalized
    w1 = 1.0
    w2 = 0.5
    w3 = 0.3
    
    scores = -w1 * deg_norm - w2 * nds_norm + w3 * v_normalized
    
    return scores
# EVOLVE-BLOCK-END
