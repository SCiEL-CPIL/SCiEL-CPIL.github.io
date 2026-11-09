"""Improved TSP constructive heuristic with look-ahead and time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for time management
_start_time = None
_alpha = 0.35  # Weight for nearest neighbor vs. insertion cost
_connectivity_weight = 0.25  # Weight for connectivity/degree-2 completion penalty
_lookahead_weight = 0.40  # Weight for 2-step lookahead


def _get_time_remaining():
    """Return seconds remaining before the 150s limit (with safety margin)."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, 140.0 - elapsed)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node in the TSP tour using a connectivity-aware heuristic.
    
    Combines:
    1. Nearest neighbor distance from current node
    2. Insertion cost (impact on closing the tour)
    3. Connectivity penalty: estimates the cost of completing the tour if we pick
       this candidate, penalizing choices that lead to high degree-2 completion costs
    4. 2-step lookahead: simulates greedy completion for 2 more steps to estimate
       the trajectory quality
    
    This prevents disconnected subtours by favoring candidates that keep the
    partial tour "connectable" with low completion cost.
    """
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    # Safety check: if we're running out of time, fall back to nearest neighbor
    if _get_time_remaining() < 0.1:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1  # Should not happen per contract
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # For very small sets, use simple nearest neighbor
    if n_unvisited <= 3:
        dist_current = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dist_current)])
    
    # Compute base scores for each unvisited node
    dist_current = distance_matrix[current_node, unvisited_nodes]  # (n_unvisited,)
    dist_dest = distance_matrix[destination_node, unvisited_nodes]  # (n_unvisited,)
    dist_current_dest = distance_matrix[current_node, destination_node]
    
    # Insertion cost: extra distance if we insert this node between current and destination
    insertion_cost = dist_current + dist_dest - dist_current_dest
    
    # Connectivity penalty: estimate the "degree-2 completion cost"
    # For each candidate, estimate how expensive it would be to complete the tour
    # This approximates: if we go current->candidate, what's the cost to eventually
    # return to destination? We use the sum of two cheapest connections from candidate
    # to other unvisited nodes as a proxy for connectivity quality.
    
    # For each candidate i, compute the sum of the 2 smallest distances from i to other unvisited nodes
    # This approximates the cost of giving candidate degree 2 in the final tour
    # Lower values mean the candidate can be easily connected to neighbors
    dist_matrix_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]  # (n, n)
    
    # For each row (candidate), find the 2 smallest distances (excluding self)
    # Set diagonal to inf so we don't pick self-connections
    dist_matrix_no_self = dist_matrix_unvisited.copy()
    np.fill_diagonal(dist_matrix_no_self, np.inf)
    
    # Get the 2 smallest values per row
    if n_unvisited >= 3:
        # Use partition for efficiency: find the 2 smallest in each row
        # np.partition gives sorted positions but not fully sorted
        two_smallest = np.partition(dist_matrix_no_self, 1, axis=1)[:, :2]
        connectivity_penalty = np.sum(two_smallest, axis=1)
    else:
        connectivity_penalty = np.sum(dist_matrix_no_self, axis=1)
    
    # 2-step lookahead: simulate greedy completion for 2 more steps
    # For each candidate, estimate the total cost of a 2-step greedy path
    lookahead_cost = np.zeros(n_unvisited)
    for i in range(n_unvisited):
        candidate = unvisited_nodes[i]
        # Step 1: from candidate, pick the nearest unvisited node (excluding self and current)
        remaining_after_i = list(unvisited_nodes)
        remaining_after_i.remove(candidate)
        remaining_after_i.remove(current_node) if current_node in remaining_after_i else None
        
        if len(remaining_after_i) >= 2:
            # Greedy step 1: nearest from candidate
            dists_from_candidate = distance_matrix[candidate, remaining_after_i]
            next1_idx = np.argmin(dists_from_candidate)
            next1 = remaining_after_i[next1_idx]
            cost1 = dists_from_candidate[next1_idx]
            
            # Greedy step 2: from next1, pick nearest remaining
            remaining_after_next1 = [x for x in remaining_after_i if x != next1]
            if len(remaining_after_next1) >= 1:
                dists_from_next1 = distance_matrix[next1, remaining_after_next1]
                next2_idx = np.argmin(dists_from_next1)
                next2 = remaining_after_next1[next2_idx]
                cost2 = dists_from_next1[next2_idx]
            else:
                cost2 = 0.0
            
            # Cost to eventually get to destination from next2
            cost3 = distance_matrix[next2, destination_node]
            
            lookahead_cost[i] = cost1 + cost2 + cost3
        elif len(remaining_after_i) == 1:
            next1 = remaining_after_i[0]
            lookahead_cost[i] = distance_matrix[candidate, next1] + distance_matrix[next1, destination_node]
        else:
            lookahead_cost[i] = distance_matrix[candidate, destination_node]
    
    # Normalize all components to [0, 1]
    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        range_val = max_val - min_val
        if range_val < 1e-10:
            return np.zeros_like(arr)
        return (arr - min_val) / range_val
    
    norm_dist = normalize(dist_current)
    norm_insert = normalize(insertion_cost)
    norm_connectivity = normalize(connectivity_penalty)
    norm_lookahead = normalize(lookahead_cost)
    
    # Combine with weights
    score = (_alpha * norm_dist + 
             (1 - _alpha - _connectivity_weight - _lookahead_weight) * norm_insert +
             _connectivity_weight * norm_connectivity +
             _lookahead_weight * norm_lookahead)
    
    best_idx = np.argmin(score)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
