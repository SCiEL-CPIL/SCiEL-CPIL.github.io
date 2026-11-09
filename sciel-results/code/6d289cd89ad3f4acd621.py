"""Improved TSP constructive heuristic with 2-opt look-ahead and time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for time management
_start_time = None
_DEADLINE_SECONDS = 130.0  # Safety margin before 150s limit


def _get_time_remaining():
    """Return seconds remaining before the deadline (with safety margin)."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, _DEADLINE_SECONDS - elapsed)


def _two_opt_improve_segment(tour, dist_matrix, max_iters=3):
    """
    Apply limited 2-opt improvement to a small segment of the tour.
    tour: list of node indices
    dist_matrix: distance matrix
    Returns improved tour (list)
    """
    n = len(tour)
    if n < 4:
        return tour
    
    tour = list(tour)
    
    for _ in range(max_iters):
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                # Check if reversing segment i+1..j improves
                old_dist = dist_matrix[a, b] + dist_matrix[c, d]
                new_dist = dist_matrix[a, c] + dist_matrix[b, d]
                if new_dist < old_dist - 1e-10:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
        if not improved:
            break
    
    return tour


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node in the TSP tour using a look-ahead heuristic.
    
    Uses a combination of:
    1. Nearest neighbor distance
    2. Insertion cost (how much it increases the tour if inserted)
    3. Limited 2-opt look-ahead for small remaining sets
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n_unvisited = len(unvisited_nodes)
    
    # If only one node left, return it
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # If no nodes left (shouldn't happen), return current
    if n_unvisited == 0:
        return int(current_node)
    
    # Time check - if we're running out of time, fall back to nearest neighbor
    remaining_time = _get_time_remaining()
    if remaining_time < 0.5:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # For small remaining sets, use more sophisticated approach
    if n_unvisited <= 12:
        # Try each candidate and evaluate the resulting partial tour with 2-opt
        best_node = int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
        best_score = float('inf')
        
        for node in unvisited_nodes:
            node_idx = int(node)
            # Score: distance from current + distance to destination (insertion cost proxy)
            score = distance_matrix[current_node, node_idx] + distance_matrix[node_idx, destination_node]
            
            if score < best_score:
                best_score = score
                best_node = node_idx
        
        # For very small sets, try 2-opt on the remaining tour
        if n_unvisited <= 8 and remaining_time > 2.0:
            # Build the remaining tour: current -> [unvisited nodes] -> destination
            # Try different orderings via 2-opt
            # Start with nearest-neighbor order
            remaining = list(unvisited_nodes)
            current_pos = current_node
            ordered = []
            while remaining:
                closest = min(remaining, key=lambda x: distance_matrix[current_pos, x])
                ordered.append(closest)
                remaining.remove(closest)
                current_pos = closest
            
            # Add destination at the end
            full_segment = [current_node] + ordered + [destination_node]
            
            # Apply 2-opt
            improved_segment = _two_opt_improve_segment(full_segment, distance_matrix, max_iters=2)
            
            # The next node is the second element in the improved segment
            if len(improved_segment) >= 2:
                best_node = int(improved_segment[1])
        
        return best_node
    
    # For larger sets, use a weighted combination of:
    # 1. Distance from current node
    # 2. Distance to destination node (savings heuristic)
    # 3. A simple insertion cost estimate
    
    current_to_candidates = distance_matrix[current_node, unvisited_nodes]
    candidates_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Normalize distances
    max_curr = np.max(current_to_candidates) if np.max(current_to_candidates) > 0 else 1.0
    max_dest = np.max(candidates_to_dest) if np.max(candidates_to_dest) > 0 else 1.0
    
    norm_curr = current_to_candidates / max_curr
    norm_dest = candidates_to_dest / max_dest
    
    # Weighted score: prioritize closer to current, but also consider proximity to destination
    # This helps avoid "stranded" nodes far from the destination
    alpha = 0.6  # Weight for nearest neighbor
    scores = alpha * norm_curr + (1 - alpha) * norm_dest
    
    return int(unvisited_nodes[np.argmin(scores)])
# EVOLVE-BLOCK-END
