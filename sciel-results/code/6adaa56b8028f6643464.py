"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline tracking
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin from 150s

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _START_TIME, _TIME_LIMIT
    
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return depot
    
    if len(unvisited_nodes) == 0:
        return depot
    
    # Convert to numpy array for fast operations
    unvisited = np.array(unvisited_nodes, dtype=np.int64)
    
    # 1. Check capacity constraints
    feasible_mask = np.array([demands[j] <= rest_capacity for j in unvisited])
    
    if not np.any(feasible_mask):
        return depot

    # Get feasible nodes
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot
    
    # Compute base distances
    dist_from_current = distance_matrix[current_node, feasible_nodes]
    dist_from_depot = distance_matrix[depot, feasible_nodes]
    
    # 2-step lookahead: for each candidate next node, estimate the best 
    # distance to another unvisited node (or depot if closing route).
    # This helps avoid choosing a node that leads to a poor next step.
    
    n_feasible = len(feasible_nodes)
    lookahead_cost = np.zeros(n_feasible)
    
    # For each candidate, compute min distance to other feasible nodes + depot
    # We want to minimize: dist(current, next) + w * min_dist(next, other)
    # where other can be other feasible nodes or depot
    
    for i in range(n_feasible):
        candidate = feasible_nodes[i]
        # Distance from candidate to all other feasible nodes
        if n_feasible > 1:
            # Exclude self
            other_dists = np.delete(distance_matrix[candidate, feasible_nodes], i)
        else:
            other_dists = np.array([])
        
        # Distance from candidate to depot (for closing route)
        dist_to_depot = distance_matrix[candidate, depot]
        
        # The best next step is min of: going to another feasible node, or going to depot
        # But we also need to consider capacity for the next node
        # For simplicity, use min distance to any other feasible node or depot
        
        if len(other_dists) > 0:
            best_next_dist = min(np.min(other_dists), dist_to_depot)
        else:
            best_next_dist = dist_to_depot
        
        lookahead_cost[i] = best_next_dist
    
    # Combine savings heuristic with lookahead
    # Savings: dist_from_depot - dist_from_current (maximize)
    # Lookahead: dist_from_current + lookahead_cost (minimize)
    
    # Weighted combination:
    # score = alpha * (dist_from_depot - dist_from_current) - beta * (dist_from_current + lookahead_cost)
    # We want to maximize this score.
    
    # Dynamic weights based on remaining capacity:
    # If capacity is low, prioritize shorter routes (higher beta)
    # If capacity is high, prioritize savings (higher alpha)
    
    # Normalize capacities: assume max capacity is around 50
    # rest_capacity / 50 gives a ratio from 0 to 1
    cap_ratio = min(1.0, rest_capacity / 50.0)
    
    # When capacity is high, trust savings more; when low, be conservative
    alpha = 0.7 + 0.3 * cap_ratio  # 0.7 to 1.0
    beta = 0.3 * (1.0 - cap_ratio) + 0.1  # 0.4 when low cap, 0.1 when high cap
    
    savings_score = dist_from_depot - dist_from_current
    lookahead_penalty = dist_from_current + lookahead_cost
    
    # Normalize both to [0, 1] for stable combination
    if np.max(savings_score) - np.min(savings_score) > 1e-10:
        savings_norm = (savings_score - np.min(savings_score)) / (np.max(savings_score) - np.min(savings_score))
    else:
        savings_norm = np.zeros(n_feasible)
    
    if np.max(lookahead_penalty) - np.min(lookahead_penalty) > 1e-10:
        lookahead_norm = (lookahead_penalty - np.min(lookahead_penalty)) / (np.max(lookahead_penalty) - np.min(lookahead_penalty))
    else:
        lookahead_norm = np.zeros(n_feasible)
    
    # Maximize: alpha * savings_norm - beta * lookahead_norm
    combined_score = alpha * savings_norm - beta * lookahead_norm
    
    best_idx = np.argmax(combined_score)
    next_node = feasible_nodes[best_idx]
    
    return int(next_node)

# EVOLVE-BLOCK-END
