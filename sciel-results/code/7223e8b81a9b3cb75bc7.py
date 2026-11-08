"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit
    
    # Convert to float matrix for efficient matrix multiplication
    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    adj_bool = adj > 0.5
    degrees = adj.sum(axis=1)
    
    # Initial order: low degree first (greedy MIS heuristic)
    # We assign initial scores so that low-degree vertices come first
    # Use negative degree as initial score
    scores = -degrees.copy()
    
    # Precompute neighbor indices for each vertex for fast access
    neighbor_indices = [np.nonzero(adj_bool[i])[0] for i in range(n)]
    
    def greedy_size_from_order(order):
        """Compute greedy MIS size given a priority order (array of vertex indices)."""
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                count += 1
                selected[v] = True
                if len(neighbor_indices[v]) > 0:
                    selected[neighbor_indices[v]] = True
        return count
    
    def order_from_scores(s):
        """Get vertex order by descending score."""
        return np.argsort(-s)
    
    # Initial greedy size
    current_order = order_from_scores(scores)
    current_size = greedy_size_from_order(current_order)
    best_size = current_size
    best_scores = scores.copy()
    
    # Local search: try swapping pairs of adjacent vertices in the order
    # If a swap improves the greedy size, keep it
    max_iterations = 500
    iter_count = 0
    
    while iter_count < max_iterations:
        if time.time() > deadline:
            break
        
        improved = False
        # Try all adjacent swaps
        for i in range(n - 1):
            if time.time() > deadline:
                break
            
            v1 = current_order[i]
            v2 = current_order[i + 1]
            
            # Swap them
            current_order[i] = v2
            current_order[i + 1] = v1
            
            new_size = greedy_size_from_order(current_order)
            
            if new_size > current_size:
                current_size = new_size
                improved = True
                if current_size > best_size:
                    best_size = current_size
                    # Update best scores to reflect this order
                    # We need to assign scores consistent with the order
                    # Use rank-based scores: higher rank = higher score
                    best_scores = np.zeros(n)
                    for rank, v in enumerate(current_order):
                        best_scores[v] = n - rank
            else:
                # Revert swap
                current_order[i] = v1
                current_order[i + 1] = v2
        
        iter_count += 1
        if not improved:
            break
    
    # If we improved, use the best order's rank-based scores
    if best_size > greedy_size_from_order(order_from_scores(-degrees)):
        # Assign scores based on the best order found
        final_order = order_from_scores(best_scores)
        # Re-verify and potentially do a few more passes
        # But the rank-based scores from best_scores should be consistent
        return best_scores.astype(float)
    
    # Otherwise, return the initial degree-based scores
    return scores.astype(float)
# EVOLVE-BLOCK-END
