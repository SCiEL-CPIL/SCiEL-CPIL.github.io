"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Assign a priority score to each vertex such that a greedy decoder
    picking vertices in descending score order (skipping neighbours of
    already-picked vertices) yields a large independent set.
    
    Strategy: Use a combination of local structural features and a 
    spectral component to capture global structure. We compute:
    1. Degree (lower is better for independent set)
    2. Number of high-degree neighbours (penalty)
    3. A spectral score based on the Fiedler vector or principal eigenvector
       of the Laplacian, which captures community/cluster structure.
    4. A refined score that accounts for the "cost" of selecting a vertex 
       in terms of how many potential independent set members it blocks.
    
    We use a fast power iteration for the spectral component to stay within
    time limits.
    """
    start_time = time.time()
    deadline = 150.0 - 5.0  # safety margin
    
    n = adjacency.shape[0]
    deg = adjacency.sum(axis=1).astype(float)
    
    # If the graph is very small or empty, handle trivially
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([1.0], dtype=float)
    
    # Convert to boolean for neighbour lookups
    adj_bool = adjacency.astype(bool)
    
    # Feature 1: Degree (lower is better)
    # Feature 2: Sum of degrees of neighbours (higher means more "crowded")
    # This is the number of edges incident to neighbours, roughly
    neigh_deg_sum = adjacency @ deg
    
    # Feature 3: Number of neighbours with degree > median (penalty for being
    # adjacent to hubs)
    median_deg = np.median(deg)
    hub_neighbours = (adjacency @ (deg > median_deg).astype(float))
    
    # Feature 4: Spectral component via power iteration on the Laplacian
    # We want to capture global structure. The Fiedler vector (second eigenvector
    # of the Laplacian) can help identify communities.
    # L = D - A
    # We'll do a few power iterations on (I - L/n) or similar to get a smooth
    # spectral score.
    
    # Actually, for independent set, we want vertices that are "central" in
    # their local structure but not in high-density clusters. Let's use the
    # principal eigenvector of the adjacency matrix as a centrality measure,
    # and combine with degree.
    
    # Power iteration for principal eigenvector of adjacency (symmetric)
    # Normalize adjacency to have spectral radius < 1
    max_deg = deg.max()
    if max_deg > 0:
        A_norm = adjacency / max_deg
    else:
        A_norm = adjacency.copy()
    
    # Start with random or degree-based initial vector
    v = np.ones(n, dtype=float)
    v /= np.linalg.norm(v)
    
    num_iters = 20  # Keep it small for speed
    for _ in range(num_iters):
        if time.time() > deadline:
            break
        v_new = A_norm @ v
        norm = np.linalg.norm(v_new)
        if norm < 1e-12:
            break
        v = v_new / norm
    
    # v is now the principal eigenvector (centrality)
    # Higher centrality means more connected, which is bad for independent set
    # So we want low centrality + low degree
    
    # Combine features:
    # score = -w1 * degree - w2 * neigh_deg_sum - w3 * hub_neighbours - w4 * centrality
    # We want to minimize this, so higher score = better candidate
    
    # Normalize each feature to [0, 1] range for stable combination
    def normalize(x):
        x_min = x.min()
        x_max = x.max()
        if x_max - x_min < 1e-12:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min)
    
    deg_norm = normalize(deg)
    neigh_norm = normalize(neigh_deg_sum)
    hub_norm = normalize(hub_neighbours)
    cent_norm = normalize(v)
    
    # Weighted combination
    # Lower degree, lower neighbour degree sum, fewer hub neighbours, lower centrality
    score = -(1.0 * deg_norm + 0.5 * neigh_norm + 0.3 * hub_norm + 0.4 * cent_norm)
    
    # Add a small random tie-breaker to avoid degeneracy
    # Use a deterministic tie-breaker based on vertex index for reproducibility
    # Actually, let's use the original index as a tiny perturbation
    # This helps when scores are equal
    
    # Return the score; higher is selected first
    return score
# EVOLVE-BLOCK-END
