"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Assign a score to each vertex such that a greedy decoder picking vertices
    in descending score order (skipping neighbors of already-picked vertices)
    yields a large independent set.
    
    Strategy: Use a combination of:
    1. Low degree (prefer isolated/low-degree vertices)
    2. Low 'blocking potential' - vertices whose neighbors have high degree
       should be deprioritized because picking them blocks fewer valuable vertices
       (actually, we want to pick vertices that block FEW other high-value vertices)
    3. Prefer vertices that are NOT adjacent to many high-degree vertices
    
    Key insight: When we pick a vertex v, we block all its neighbors.
    To maximize the independent set size, we should prefer vertices v where:
    - v has low degree (blocks few neighbors)
    - v's neighbors have LOW degree (so blocking them doesn't remove many
      potential future picks)
    
    A good heuristic score: 
    score(v) = -degree(v) - sum_{u in N(v)} degree(u) / (some normalization)
    
    But even better: think about what the greedy algorithm actually does.
    If we pick v first, we lose all neighbors of v. The "cost" of picking v is
    roughly the number of neighbors, but also the opportunity cost of those
    neighbors not being picked later.
    
    A refined approach: 
    score(v) = -degree(v) - alpha * sum_{u in N(v)} degree(u)
    
    where alpha balances direct blocking vs. indirect opportunity cost.
    
    Let's use: score(v) = -(degree(v) + 0.5 * sum_of_neighbor_degrees(v))
    
    This penalizes vertices that have many neighbors AND whose neighbors also
    have many neighbors (dense clusters).
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin within 120s limit
    
    # Convert to numpy array if needed
    adj = np.asarray(adjacency)
    n = adj.shape[0]
    
    # Compute degrees
    degrees = adj.sum(axis=1).astype(float)
    
    # Compute sum of neighbor degrees for each vertex
    # sum_of_neighbor_degrees[v] = sum of degrees of all neighbors of v
    # This is: degrees @ adj (since adj is symmetric for undirected graphs)
    # Actually: sum_{u in N(v)} deg(u) = (adj @ degrees)[v]
    sum_neighbor_degrees = adj.dot(degrees)
    
    # Time check
    if time.time() > deadline:
        # Fallback: just use negative degree
        return -degrees
    
    # Combine: penalize high degree and high sum of neighbor degrees
    # The coefficient on sum_neighbor_degrees controls how much we care about
    # the "quality" of the neighbors we're blocking
    alpha = 0.5
    
    scores = -(degrees + alpha * sum_neighbor_degrees)
    
    # Time check
    if time.time() > deadline:
        return scores
    
    # Optional: Add a small refinement using second-order information
    # For vertices with very high "blocking cost", further penalize
    # This helps in dense regions
    
    # Check if we have time for a more sophisticated approach
    if time.time() < deadline - 5:
        # Try to add a clustering coefficient-like penalty
        # Count triangles: for each vertex, count edges among its neighbors
        # This is expensive, so only do it if n is small enough
        if n <= 500:
            # Triangle count approximation using matrix multiplication
            # For each vertex v, the number of edges among N(v) is:
            # (adj[neighbors_of_v] @ adj[neighbors_of_v].T) diagonal / 2
            # But this is O(n^3) in worst case. Let's use a simpler proxy.
            
            # Alternative: use the "local density" as sum of degrees of neighbors
            # divided by degree (average neighbor degree)
            # Vertices with high average neighbor degree are in dense regions
            avg_neighbor_degree = np.zeros(n)
            nonzero_deg = degrees > 0
            avg_neighbor_degree[nonzero_deg] = sum_neighbor_degrees[nonzero_deg] / degrees[nonzero_deg]
            
            # Penalize vertices in dense regions more heavily
            # But be careful: a vertex with high degree in a sparse graph might be fine
            # Let's add a mild penalty based on average neighbor degree
            beta = 0.1
            scores = scores - beta * avg_neighbor_degree
            
            if time.time() > deadline:
                return scores
    
    return scores
# EVOLVE-BLOCK-END
