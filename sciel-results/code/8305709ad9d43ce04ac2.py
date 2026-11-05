"""Improved program for Maximum Independent Set via structural isolation scoring + local swap refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_iset_size(adjacency, order):
    """Compute the size of the independent set obtained by greedy selection in given order."""
    n = adjacency.shape[0]
    selected = np.zeros(n, dtype=bool)
    count = 0
    for v in order:
        if not selected[v]:
            selected[v] = True
            count += 1
            # Mark neighbors as selected (blocked)
            neighbors = np.nonzero(adjacency[v])[0]
            selected[neighbors] = True
    return count


def priority(adjacency):
    """
    Assign scores to vertices to maximize the greedy independent set size.
    
    Strategy:
    1. Compute a structural isolation score based on inverse weighted degree in non-neighborhood.
    2. Use this as an initial ordering.
    3. Perform local swap improvements to refine the ordering.
    
    Returns: array of scores (higher = more likely to be selected early).
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    # --- Step 1: Initial scoring based on structural isolation ---
    # Degree of each vertex
    degrees = adj.sum(axis=1).astype(float)
    
    # For each vertex, compute weighted degree in its non-neighborhood
    # This measures how "isolated" a vertex is from dense clusters
    # A vertex with low weighted degree in its non-neighbors is structurally isolated
    
    # Non-neighborhood: for vertex i, vertices j where adj[i,j] == 0 and j != i
    # Weighted degree in non-neighborhood = sum of degrees of non-neighbors
    # This is expensive for large n, so use an approximation
    
    # Approximation: use the sum of degrees of non-neighbors
    # For each vertex i, non_neighbor_degree_sum[i] = sum_{j: adj[i,j]==0, j!=i} degrees[j]
    
    # Compute efficiently:
    # total_degree_sum = sum of all degrees
    # neighbor_degree_sum[i] = sum of degrees of neighbors of i
    # non_neighbor_degree_sum[i] = total_degree_sum - neighbor_degree_sum[i] - degrees[i]
    
    total_degree_sum = degrees.sum()
    
    # neighbor_degree_sum[i] = sum of degrees of neighbors of i
    # This is adj @ degrees
    neighbor_degree_sum = adj.astype(float) @ degrees
    
    # non_neighbor_degree_sum[i] = total_degree_sum - neighbor_degree_sum[i] - degrees[i]
    non_neighbor_degree_sum = total_degree_sum - neighbor_degree_sum - degrees
    
    # Ensure non-negative
    non_neighbor_degree_sum = np.maximum(non_neighbor_degree_sum, 0)
    
    # Structural isolation score: higher means more isolated
    # Use inverse of (1 + non_neighbor_degree_sum) to avoid division by zero
    # Also incorporate the vertex's own degree (lower degree = easier to include)
    isolation_score = 1.0 / (1.0 + non_neighbor_degree_sum)
    
    # Combine with inverse degree
    # Vertices with low degree and high isolation should be prioritized
    combined_score = isolation_score * (1.0 / (1.0 + degrees))
    
    # Normalize to [0, 1]
    min_s = combined_score.min()
    max_s = combined_score.max()
    if max_s > min_s:
        combined_score = (combined_score - min_s) / (max_s - min_s)
    else:
        combined_score = np.zeros(n, dtype=float)
    
    # Initial ordering: descending score
    order = np.argsort(-combined_score)
    
    # --- Step 2: Local swap improvement ---
    # Try swapping adjacent pairs in the ordering and keep if it improves the IS size
    current_size = _greedy_iset_size(adj, order)
    best_order = order.copy()
    best_size = current_size
    
    # Limit the number of swap iterations based on time
    max_swap_rounds = 50
    improved = True
    round_num = 0
    
    while improved and round_num < max_swap_rounds:
        improved = False
        round_num += 1
        
        # Check time budget
        if time.time() > deadline - 10.0:
            break
        
        for i in range(n - 1):
            if time.time() > deadline - 10.0:
                break
            
            # Try swapping positions i and i+1
            new_order = best_order.copy()
            new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]
            
            new_size = _greedy_iset_size(adj, new_order)
            
            if new_size > best_size:
                best_size = new_size
                best_order = new_order
                improved = True
    
    # --- Step 3: Final scoring based on refined ordering ---
    # Assign scores such that vertices earlier in best_order have higher scores
    # Use rank-based scoring
    ranks = np.zeros(n, dtype=float)
    for idx, v in enumerate(best_order):
        ranks[v] = n - idx  # Higher score for earlier position
    
    return ranks
# EVOLVE-BLOCK-END
