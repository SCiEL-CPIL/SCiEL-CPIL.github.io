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
    
    # Let's compute this cost.
    dist_from_current = distance_matrix[current_node, feasible_nodes]
    dist_from_depot = distance_matrix[depot, feasible_nodes]
    
    # Cost = dist_from_current - dist_from_depot
    # We want to minimize this cost.
    # Note: dist_from_depot is not fixed for all nodes, so we can't drop it.
    # dist_from_current varies, dist_from_depot varies.
    
    # Let's just minimize: dist_from_current - dist_from_depot
    costs = dist_from_current - dist_from_depot
    
    # However, if a node is very far from current, even if it's far from depot, it might be bad.
    # Let's check if pure savings is better or worse than nearest.
    # The previous program used pure savings (max dist_depot - dist_current) and scored 1.082.
    # The seed (nearest) scored 1.0.
    # So pure savings was an improvement? 1.082 > 1.0 means better?
    # "Score: mean total route length ... (lower is better)"
    # "search score is seed/value"
    # If seed is 1.0 and value is 1.0, score is 1.0.
    # If value is lower (better), score is higher.
    # So 1.082 is better than 1.0.
    # Therefore, the savings heuristic improved upon nearest neighbor.
    
    # Can we do even better?
    # Maybe add a small tie-breaker or adjust for capacity?
    # If we have lots of capacity, maybe we can afford to go a bit further to get a big saving.
    # If we have little capacity, we should maybe grab the nearest feasible node to close the route quickly?
    # Or grab the node that is closest to the depot to make the return trip short?
    
    # Let's try to incorporate capacity.
    # If rest_capacity is very low, prioritize nodes that are close to the depot (so we can return soon).
    # This would mean minimizing dist_from_depot?
    # But we also need to get there.
    
    # Pure Savings heuristic was better (1.082) than the hybrid (1.038).
    # Maximize: dist(depot, next) - dist(current, next)
    # This is equivalent to minimizing: dist(current, next) - dist(depot, next)
    
    # Dynamic penalty: when capacity is low, we should prefer nodes that are
    # closer to the depot (shorter return trip) and penalize long detours more.
    # Normalize rest_capacity to [0, 1] assuming max demand ~9, capacity 50.
    # As rest_capacity decreases, increase the weight on dist_from_depot penalty.
    
    # Estimate fraction of capacity used (0 = full capacity left, 1 = almost empty)
    # We don't know total capacity, but we can use rest_capacity relative to typical values.
    # Use a smooth function: penalty_weight = 1.0 / (1.0 + rest_capacity / 10.0)
    # When rest_capacity is high (e.g., 50), penalty_weight ~ 0.17 (small penalty)
    # When rest_capacity is low (e.g., 5), penalty_weight ~ 0.5 (larger penalty)
    # When rest_capacity is very low (e.g., 1), penalty_weight ~ 0.91 (strong penalty)
    
    penalty_weight = 1.0 / (1.0 + rest_capacity / 10.0)
    
    # Also add a small bias toward nodes with lower demand when capacity is tight,
    # to preserve capacity for more nodes. But this is secondary.
    
    # Base savings score: maximize (dist_depot - dist_current)
    # Penalty: minimize dist_from_depot (prefer nodes close to depot when capacity is low)
    # Combined: maximize (dist_depot - dist_current) - penalty_weight * dist_from_depot
    #          = maximize (1 - penalty_weight) * dist_depot - dist_current
    
    alpha = 1.0 - penalty_weight  # Weight on dist_depot in the savings term
    scores = alpha * dist_from_depot - dist_from_current
    
    # We want to maximize scores.
    best_idx = np.argmax(scores)
    next_node = feasible_nodes[best_idx]
    
    return int(next_node)

# EVOLVE-BLOCK-END
