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
    n_total = len(unvisited)
    
    # 1. Check capacity constraints
    feasible_mask = np.array([demands[j] <= rest_capacity for j in unvisited])
    
    if not np.any(feasible_mask):
        return depot

    # Get feasible nodes
    feasible_nodes = unvisited[feasible_mask]
    n_feasible = len(feasible_nodes)
    
    if n_feasible == 0:
        return depot
    
    if n_feasible == 1:
        return int(feasible_nodes[0])
    
    # Compute distances from current node and depot to all feasible nodes
    dist_from_current = distance_matrix[current_node, feasible_nodes].astype(np.float64)
    dist_from_depot = distance_matrix[depot, feasible_nodes].astype(np.float64)
    
    # Compute pairwise distances among feasible nodes for regret calculation
    # This is O(n_feasible^2) but n_feasible is at most 50, so it's fast
    # We need for each node i: min_{j != i} dist(i, j) where dist is from current node
    # Actually, the standard regret-2 considers: for node i, 
    #   regret_i = (second best distance from current to any node) - (best distance from current to i)
    # But a more useful variant for routing:
    #   For each candidate node i, consider the cost of NOT visiting i now.
    #   If we don't visit i now, we'll have to visit it later, and the best "partner" 
    #   to visit with it would be the node j that minimizes dist(current, j) + dist(j, i) - dist(current, i)
    #   Actually, let's use a simpler and effective regret:
    #   regret_i = min_{j != i} dist_from_current[j] - dist_from_current[i]
    #   This measures how much worse the next-best option is compared to visiting i now.
    
    # Compute regret-2: for each node i, the difference between the 2nd smallest and smallest 
    # distance from current among all feasible nodes, but specifically:
    #   regret_i = (min_{j != i} dist_from_current[j]) - dist_from_current[i]
    # If regret_i is large and positive, node i is uniquely close to current, so we should visit it now.
    
    sorted_indices = np.argsort(dist_from_current)
    min_val = dist_from_current[sorted_indices[0]]
    second_min_val = dist_from_current[sorted_indices[1]]
    
    min_excluding_self = np.full(n_feasible, min_val, dtype=np.float64)
    min_excluding_self[sorted_indices[0]] = second_min_val
    
    regret = min_excluding_self - dist_from_current
    # Clip negative regrets to 0
    regret = np.maximum(regret, 0.0)
    
    # Normalize regret to [0, 1]
    max_regret = np.max(regret)
    if max_regret > 1e-12:
        regret_normalized = regret / max_regret
    else:
        regret_normalized = np.zeros(n_feasible)
    
    # Savings score: dist(depot, i) - dist(current, i)
    # High savings means visiting i now saves us from having to go from depot to i later
    savings_score = dist_from_depot - dist_from_current
    
    # Normalize savings to [0, 1]
    max_savings = np.max(savings_score)
    min_savings = np.min(savings_score)
    if max_savings - min_savings > 1e-12:
        savings_normalized = (savings_score - min_savings) / (max_savings - min_savings)
    else:
        savings_normalized = np.zeros(n_feasible)
    
    # Dynamic weighting based on remaining capacity
    total_capacity = 50.0
    capacity_ratio = rest_capacity / total_capacity
    
    # When capacity is low, we have fewer choices, so regret matters more
    # When capacity is high, we can afford to optimize for savings
    if capacity_ratio < 0.25:
        # Very low capacity: prioritize regret heavily
        regret_weight = 0.65
        savings_weight = 0.35
    elif capacity_ratio < 0.5:
        # Medium-low capacity: balance with slight regret preference
        regret_weight = 0.45
        savings_weight = 0.55
    else:
        # High capacity: prioritize savings
        regret_weight = 0.25
        savings_weight = 0.75
    
    # Combined score
    combined_score = savings_weight * savings_normalized + regret_weight * regret_normalized
    
    # Add a small penalty for very long distances to avoid extreme detours
    max_dist = np.max(dist_from_current) if n_feasible > 0 else 1.0
    if max_dist > 1e-12:
        dist_penalty = 0.08 * (dist_from_current / max_dist)
        combined_score = combined_score - dist_penalty
    
    # We want to maximize the combined score
    best_idx = np.argmax(combined_score)
    next_node = feasible_nodes[best_idx]
    
    return int(next_node)

# EVOLVE-BLOCK-END
