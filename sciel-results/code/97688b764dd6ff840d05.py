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
    
    n_feasible = len(feasible_nodes)
    
    if n_feasible <= 1:
        return int(feasible_nodes[0])
    
    # Compute distances
    dist_from_current = distance_matrix[current_node, feasible_nodes].astype(np.float64)
    dist_from_depot = distance_matrix[depot, feasible_nodes].astype(np.float64)
    
    # Compute savings score: maximize dist(depot, next) - dist(current, next)
    savings_score = dist_from_depot - dist_from_current
    
    # Compute regret: for each node i, regret_i = min_j!=i dist_from_current[j] - dist_from_current[i]
    # High regret means node i is uniquely close and should be visited now
    # Efficient computation using sorting
    sorted_indices = np.argsort(dist_from_current)
    
    # For each position in sorted order, the minimum excluding self is:
    # - If at position 0 (smallest), the min excluding self is the second smallest (position 1)
    # - If at position k > 0, the min excluding self is the smallest (position 0)
    min_excluding_self = np.empty(n_feasible, dtype=np.float64)
    
    # The smallest distance is at sorted_indices[0]
    # For the node at sorted_indices[0], min excluding self is dist_from_current[sorted_indices[1]]
    # For all other nodes, min excluding self is dist_from_current[sorted_indices[0]]
    
    min_val = dist_from_current[sorted_indices[0]]
    second_min_val = dist_from_current[sorted_indices[1]]
    
    # Initialize all to min_val (the global minimum)
    min_excluding_self[:] = min_val
    # For the node that IS the global minimum, use second_min_val
    min_excluding_self[sorted_indices[0]] = second_min_val
    
    # Regret: positive means node i is closer to current than any alternative
    regret = min_excluding_self - dist_from_current
    
    # Normalize regret to [0, 1] for combination
    max_regret = np.max(regret)
    min_regret = np.min(regret)
    if max_regret > min_regret:
        regret_normalized = (regret - min_regret) / (max_regret - min_regret)
    else:
        regret_normalized = np.zeros(n_feasible)
    
    # Normalize savings to [0, 1]
    max_savings = np.max(savings_score)
    min_savings = np.min(savings_score)
    if max_savings > min_savings:
        savings_normalized = (savings_score - min_savings) / (max_savings - min_savings)
    else:
        savings_normalized = np.zeros(n_feasible)
    
    # Dynamic weighting: when capacity is low, weight regret more heavily
    # to lock in uniquely positioned nodes before we're forced to close the route
    total_capacity = 50
    capacity_ratio = rest_capacity / total_capacity
    
    # When capacity is low, prioritize high-regret nodes more
    if capacity_ratio < 0.3:
        regret_weight = 0.4 + 0.3 * (0.3 - capacity_ratio) / 0.3  # 0.4 to 0.7
        savings_weight = 1.0 - regret_weight
    else:
        regret_weight = 0.25
        savings_weight = 0.75
    
    # Combined score
    combined_score = savings_weight * savings_normalized + regret_weight * regret_normalized
    
    # Add a small penalty for very long distances to avoid extreme detours
    max_dist = np.max(dist_from_current) if len(dist_from_current) > 0 else 1.0
    if max_dist > 0:
        dist_penalty = 0.05 * (dist_from_current / max_dist)
        combined_score = combined_score - dist_penalty
    
    # We want to maximize the combined score
    best_idx = np.argmax(combined_score)
    next_node = feasible_nodes[best_idx]
    
    return int(next_node)

# EVOLVE-BLOCK-END
