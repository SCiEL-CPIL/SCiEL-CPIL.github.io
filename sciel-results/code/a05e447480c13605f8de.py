"""Traveling salesman constructive heuristic with time-aware lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard: 150s wall-clock limit, use 145s for safety
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # seconds


def _time_exceeded():
    return (time.time() - _START_TIME) > _TIME_LIMIT


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next city for a TSP tour being built step-by-step.
    
    Uses a hybrid strategy:
    - For small remaining sets, use 2-opt lookahead (evaluate partial tour extension)
    - For larger sets, use a smoothed nearest-neighbor with destination bias
    - Always respects the time deadline
    
    Parameters:
        current_node: int - current position in the tour
        destination_node: int - the final node to return to (start of tour)
        unvisited_nodes: np.ndarray - array of unvisited node indices
        distance_matrix: np.ndarray - NxN distance matrix
    
    Returns:
        int - the next node to visit
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        # Should not happen, but guard against it
        return int(destination_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Time check: if we're close to deadline, fall back to simple NN
    if _time_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Get distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from all unvisited nodes to destination (for closing the tour)
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Strategy based on problem size
    if n_unvisited <= 12:
        # Small enough for 1-step lookahead: pick node that minimizes
        # dist(current, candidate) + min_{other in unvisited\{candidate}} dist(candidate, other)
        # This is a simple 2-step lookahead
        best_score = np.inf
        best_idx = 0
        
        for i in range(n_unvisited):
            # Cost to go from current to this candidate
            cost_to_candidate = dist_from_current[i]
            
            # If this is the last unvisited node, cost is just the return to destination
            if n_unvisited == 2:
                total_cost = cost_to_candidate + dist_to_dest[i]
            else:
                # Cost to go from candidate to its nearest other unvisited neighbor
                # (excluding itself)
                other_indices = np.concatenate([unvisited_nodes[:i], unvisited_nodes[i+1:]])
                if len(other_indices) > 0:
                    min_next_dist = np.min(distance_matrix[unvisited_nodes[i], other_indices])
                else:
                    min_next_dist = 0.0
                
                total_cost = cost_to_candidate + min_next_dist + dist_to_dest[i] * 0.3
                # Weight the return-to-destination cost lightly to encourage
                # choosing nodes that are also close to the destination
        
            if total_cost < best_score:
                best_score = total_cost
                best_idx = i
        
        return int(unvisited_nodes[best_idx])
    
    elif n_unvisited <= 30:
        # Medium: use a soft-min based scoring that balances
        # proximity to current and proximity to destination
        # Score = dist_from_current + alpha * softmin_dist_to_other_unvisited + beta * dist_to_dest
        
        # Compute pairwise distances among unvisited nodes for soft-min
        # But this is O(n^2) which for n=30 is 900 ops, fine
        
        # Simple approach: use a weighted combination
        # Normalize distances to [0,1] range for stable weighting
        max_dist = np.max(dist_from_current) if np.max(dist_from_current) > 0 else 1.0
        min_dist = np.min(dist_from_current)
        range_dist = max_dist - min_dist if max_dist > min_dist else 1.0
        
        normalized_current = (dist_from_current - min_dist) / range_dist
        normalized_dest = dist_to_dest / (np.max(dist_to_dest) if np.max(dist_to_dest) > 0 else 1.0)
        
        # Weighted score: primarily nearest neighbor, with a pull toward destination
        score = 0.7 * normalized_current + 0.3 * normalized_dest
        
        # Add a small penalty for nodes that are far from their nearest unvisited neighbor
        # This helps avoid creating long edges
        if n_unvisited > 2:
            # For each candidate, find min distance to other unvisited
            min_neighbor_dists = np.zeros(n_unvisited)
            for i in range(n_unvisited):
                others = np.concatenate([unvisited_nodes[:i], unvisited_nodes[i+1:]])
                if len(others) > 0:
                    min_neighbor_dists[i] = np.min(distance_matrix[unvisited_nodes[i], others])
            
            max_min_neighbor = np.max(min_neighbor_dists) if np.max(min_neighbor_dists) > 0 else 1.0
            normalized_neighbor = min_neighbor_dists / max_min_neighbor
            
            score = score + 0.15 * normalized_neighbor
        
        return int(unvisited_nodes[np.argmin(score)])
    
    else:
        # Large: use efficient vectorized scoring
        # Primary: nearest neighbor
        # Secondary: bias toward destination to avoid long closing edge
        
        # Compute a combined score
        # Use a temperature-based soft selection that gets more greedy as we approach deadline
        
        elapsed = time.time() - _START_TIME
        remaining_time = _TIME_LIMIT - elapsed
        # As time runs out, become more purely nearest-neighbor
        if remaining_time < 10.0:
            # Pure NN
            return int(unvisited_nodes[np.argmin(dist_from_current)])
        
        # Weighted score: NN with destination bias
        # The bias should be stronger when fewer nodes remain
        # (but this branch is for >30, so moderate bias)
        
        max_current = np.max(dist_from_current) if np.max(dist_from_current) > 0 else 1.0
        max_dest = np.max(dist_to_dest) if np.max(dist_to_dest) > 0 else 1.0
        
        # Scale the destination weight: stronger pull when we have many nodes left
        # because we want to keep the option of returning to destination open
        dest_weight = 0.2
        
        score = dist_from_current / max_current + dest_weight * dist_to_dest / max_dest
        
        return int(unvisited_nodes[np.argmin(score)])
# EVOLVE-BLOCK-END
