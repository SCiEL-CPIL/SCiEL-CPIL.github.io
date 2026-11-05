import numpy as np
import time

def priority(adjacency):
    """
    Compute priority scores for vertices in a graph.
    Higher score = more likely to be selected in the greedy independent set.
    
    Strategy: Hub-avoidance weighted scoring.
    - Penalize vertices adjacent to high-degree hubs
    - Reward vertices whose neighbors are low-degree (safe to pick)
    - Incorporate second-order neighborhood structure
    """
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s limit
    
    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    
    # Degree of each vertex
    degree = adj.sum(axis=1)
    
    # Avoid division by zero
    degree_safe = np.where(degree > 0, degree, 1.0)
    
    # Inverse degree of each vertex (high for low-degree vertices)
    inv_degree = 1.0 / degree_safe
    
    # Sum of inverse degrees of neighbors for each vertex
    # This captures: if my neighbors are all low-degree, I'm "safe" to pick
    neighbor_inv_degree_sum = adj @ inv_degree
    
    # Sum of degrees of neighbors (local density around each vertex)
    neighbor_degree_sum = adj @ degree
    
    # Second-order: sum of inverse degrees of neighbors' neighbors
    # This captures broader structural context
    # inv_degree_of_neighbors = adj @ inv_degree (already computed)
    # Then for each vertex, sum inv_degree of its neighbors' neighbors
    # This is adj @ (adj @ inv_degree)
    second_order_inv = adj @ neighbor_inv_degree_sum
    
    # Build composite score:
    # 1. Base: negative degree (prefer low-degree vertices)
    # 2. Bonus: high neighbor_inv_degree_sum (neighbors are low-degree, so picking me doesn't waste high-value hubs)
    # 3. Penalty: high neighbor_degree_sum (I'm in a dense area, picking me blocks many)
    # 4. Second-order bonus: prefer vertices whose neighbor-neighborhood is sparse
    
    # Normalize components to comparable scales
    deg_norm = degree / (degree.max() + 1e-9) if degree.max() > 0 else degree
    nis_norm = neighbor_inv_degree_sum / (neighbor_inv_degree_sum.max() + 1e-9) if neighbor_inv_degree_sum.max() > 0 else neighbor_inv_degree_sum
    nds_norm = neighbor_degree_sum / (neighbor_degree_sum.max() + 1e-9) if neighbor_degree_sum.max() > 0 else neighbor_degree_sum
    so_norm = second_order_inv / (second_order_inv.max() + 1e-9) if second_order_inv.max() > 0 else second_order_inv
    
    # Composite score: we want HIGH score for vertices we want to pick
    # - Low degree is good: -deg_norm
    # - Neighbors being low-degree is good: +nis_norm
    # - Being in dense area is bad: -nds_norm
    # - Sparse second-neighborhood is good: +so_norm
    
    score = (-1.0 * deg_norm + 
             1.5 * nis_norm + 
             -0.5 * nds_norm + 
             0.5 * so_norm)
    
    # Add small tie-breaking based on vertex index to ensure deterministic ordering
    # but keep it very small so it doesn't affect the main scoring
    score += np.arange(n) * 1e-10
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Shouldn't happen for this computation, but guard just in case
        pass
    
    return score.astype(np.float64)
