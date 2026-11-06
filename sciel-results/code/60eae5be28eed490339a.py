"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline tracking
_start_time = None


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    
    # Initialize deadline tracking
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: return depot if we've exceeded 145s
    elapsed = time.time() - _start_time
    if elapsed > 145:
        return depot
    
    if len(unvisited_nodes) == 0:
        return depot
    
    unvisited = np.array(unvisited_nodes, dtype=int)
    n = len(unvisited)
    
    if n == 0:
        return depot
    
    # Get distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited]
    
    # Get demands of unvisited nodes
    unvisited_demands = demands[unvisited]
    
    # Feasibility: demand must fit in remaining capacity
    feasible_mask = unvisited_demands <= rest_capacity
    
    if not np.any(feasible_mask):
        return depot
    
    feasible_indices = np.where(feasible_mask)[0]
    feasible_nodes = unvisited[feasible_indices]
    feasible_dist = dist_from_current[feasible_indices]
    feasible_demands = unvisited_demands[feasible_indices]
    
    if len(feasible_nodes) == 0:
        return depot
    
    # Compute distance from each feasible candidate to depot
    dist_to_depot = distance_matrix[feasible_nodes, depot]
    
    # 2-step lookahead: for each feasible node j, find the best next node k (from remaining unvisited)
    # that minimizes dist[j->k] + dist[k->depot], subject to capacity feasibility
    # This approximates the cost of committing to j now
    
    # For efficiency, compute dist matrix among feasible nodes + depot
    # dist[j, k] for j, k in feasible_nodes
    dist_feasible = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    
    # For each j, compute the minimum over k of (dist[j,k] + dist[k,depot]) 
    # where k is feasible after visiting j (i.e., demand[k] <= rest_capacity - demand[j])
    # Simplification: use all feasible nodes as potential next stops (relaxation)
    
    # The 2-step cost for choosing j: dist[current->j] + min_k(dist[j->k] + dist[k->depot])
    # But we should also consider going directly to depot after j if no good k exists
    
    # Compute min_next_cost[j] = min over k of (dist[j,k] + dist[k,depot])
    # with constraint that demand[k] <= rest_capacity - demand[j]
    
    n_feasible = len(feasible_nodes)
    min_next_cost = np.full(n_feasible, dist_to_depot)  # default: go to depot after j
    
    for j_idx in range(n_feasible):
        remaining_cap = rest_capacity - feasible_demands[j_idx]
        # Find k's that fit in remaining capacity
        valid_k_mask = feasible_demands <= remaining_cap
        if not np.any(valid_k_mask):
            continue
        # Get dist from j to all valid k's
        dist_j_to_k = dist_feasible[j_idx, valid_k_mask]
        dist_k_to_depot = dist_to_depot[valid_k_mask]
        next_cost = dist_j_to_k + dist_k_to_depot
        min_cost = np.min(next_cost)
        # Also consider going directly to depot
        if min_cost < min_next_cost[j_idx]:
            min_next_cost[j_idx] = min_cost
    
    # Combined score: dist[current->j] + min_next_cost[j]
    # This is a 2-step lookahead that balances immediate distance with future route efficiency
    scores = feasible_dist + min_next_cost
    
    # Add a small Clarke-Wright savings component to encourage route consolidation
    # Savings of visiting j now vs later: dist[current->depot] + dist[depot->j] - dist[current->j]
    dist_current_to_depot = distance_matrix[current_node, depot]
    savings = dist_current_to_depot + dist_to_depot - feasible_dist
    # Normalize savings to [0,1] range for blending
    if np.max(savings) > np.min(savings):
        savings_norm = (savings - np.min(savings)) / (np.max(savings) - np.min(savings))
    else:
        savings_norm = np.zeros_like(savings)
    
    # Blend: primarily use 2-step lookahead, with small savings boost
    alpha_savings = 0.1
    final_scores = scores - alpha_savings * savings_norm * (np.max(scores) - np.min(scores) + 1e-9)
    
    # Pick the node with minimum final score
    best_idx = np.argmin(final_scores)
    
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
