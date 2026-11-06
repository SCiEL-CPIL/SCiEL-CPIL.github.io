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
    # demands is a list/array. Ensure it's aligned with node indices.
    # The interface implies demands[i] is demand of node i.
    feasible_mask = np.array([demands[j] <= rest_capacity for j in unvisited])
    
    if not np.any(feasible_mask):
        return depot

    # Get feasible nodes
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot
    
    # 2. Calculate a composite score
    # We want to minimize total route length.
    # Heuristics:
    # a) Nearest Neighbor: Minimize distance from current node to next node.
    # b) Savings: Maximize (dist(depot, current) + dist(depot, next) - dist(current, next))
    #    This is the classic Clarke-Wright savings heuristic adapted for sequential choice.
    #    Since dist(depot, current) is fixed, we maximize (dist(depot, next) - dist(current, next)).
    # c) Load balancing: If we have very little capacity left, prefer nodes with high demand? 
    #    Or if we have lots of capacity, prefer nodes that are "expensive" to serve from depot?
    
    # Let's try a weighted combination.
    # dist_current = distance_matrix[current_node, feasible_nodes]
    # dist_depot = distance_matrix[depot, feasible_nodes]
    
    # Savings score: dist_depot - dist_current
    # Higher is better for savings.
    
    # Distance score: dist_current
    # Lower is better for distance.
    
    # Let's normalize scores to [0, 1] to combine them?
    # Or just use a simple linear combination.
    
    # Option 1: Pure Savings
    # score = dist_depot - dist_current
    
    # Option 2: Nearest Neighbor
    # score = -dist_current
    
    # Option 3: Hybrid
    # score = alpha * (dist_depot - dist_current) + (1-alpha) * (-dist_current)
    #       = alpha * dist_depot - alpha * dist_current - (1-alpha) * dist_current
    #       = alpha * dist_depot - dist_current
    
    # So if alpha=1, it's pure savings (dist_depot - dist_current).
    # If alpha=0, it's pure nearest (-dist_current).
    
    # Let's try a dynamic alpha.
    # If rest_capacity is low, we should be careful about going far? Or maybe pick the biggest demand that fits?
    # Actually, if capacity is low, we might be forced to close the route soon.
    # Picking the nearest node keeps us close to depot? No, it keeps us close to current.
    # Picking the node with high dist_depot (far from depot) might be good if we can serve it now.
    
    # Let's stick to a robust heuristic: Savings with a penalty for distance.
    # Minimize: dist(current, next) - lambda * (dist(depot, next) - dist(depot, current))
    # Wait, the savings term is (dist(depot, current) + dist(depot, next) - dist(current, next)).
    # Maximizing savings is equivalent to minimizing: dist(current, next) - dist(depot, next) (since dist(depot, current) is const).
    
    # So minimize: dist(current, next) - dist(depot, next)
    # This is the same as minimizing: - (dist(depot, next) - dist(current, next))
    
    # Compute distances
    dist_from_current = distance_matrix[current_node, feasible_nodes]
    dist_from_depot = distance_matrix[depot, feasible_nodes]
    
    # Base savings score: maximize dist(depot, next) - dist(current, next)
    # This encourages visiting nodes where the detour is small relative to the depot distance
    savings_score = dist_from_depot - dist_from_current
    
    # Dynamic adjustment based on remaining capacity
    # When capacity is low, we should be more conservative and prefer nodes closer to depot
    # to minimize the return trip cost
    total_capacity = 50  # Given in problem statement
    capacity_ratio = rest_capacity / total_capacity
    
    # When capacity is low (below 20%), add a penalty for nodes far from depot
    # This encourages closing the route sooner with nodes that are easy to return from
    if capacity_ratio < 0.2:
        # Scale the penalty: stronger when capacity is very low
        penalty_weight = (0.2 - capacity_ratio) / 0.2  # 0 to 1
        depot_proximity_bonus = penalty_weight * 0.3 * dist_from_depot
        savings_score = savings_score + depot_proximity_bonus
    
    # Add a small distance penalty to avoid extremely long detours
    # This prevents choosing a node with high savings but very long travel distance
    max_dist = np.max(dist_from_current) if len(dist_from_current) > 0 else 1.0
    if max_dist > 0:
        dist_penalty = 0.1 * (dist_from_current / max_dist)
        savings_score = savings_score - dist_penalty
    
    # We want to maximize the adjusted savings score
    best_idx = np.argmax(savings_score)
    next_node = feasible_nodes[best_idx]
    
    return int(next_node)

# EVOLVE-BLOCK-END
