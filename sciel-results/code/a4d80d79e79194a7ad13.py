"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline guard and caching
_start_time = None
_DEADLINE = 150.0  # Wall-clock limit
_SAFETY_MARGIN = 5.0  # Seconds to spare before deadline

def _init_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()

def _time_left():
    if _start_time is None:
        return _DEADLINE - _SAFETY_MARGIN
    return _DEADLINE - _SAFETY_MARGIN - (time.time() - _start_time)

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic to select the next customer or return to depot.
    
    Strategy:
    1. If no unvisited nodes, return to depot.
    2. Filter unvisited nodes to those that fit in remaining capacity.
    3. If no nodes fit, return to depot.
    4. Use a combined score: balance between distance to node, 
       distance from node to depot (or nearest remaining), and demand.
    
    The score for each candidate node i is:
    score(i) = dist(current, i) + alpha * dist(i, depot) + beta * demand(i)
    
    We pick the node with minimum score. This balances:
    - Short distance from current position
    - Short distance to return to depot (reducing future route length)
    - Low demand (preserving capacity for more customers)
    """
    _init_time()
    
    # Check deadline - if running out of time, use fast greedy
    if _time_left() < 0.01:
        # Emergency fallback: nearest feasible
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return 0
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])
    
    # If no unvisited nodes, return to depot
    if len(unvisited_nodes) == 0:
        return 0
    
    # Filter to feasible nodes (demand fits in remaining capacity)
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    
    # If no feasible nodes, return to depot
    if len(feasible) == 0:
        return 0
    
    # Compute scores for each feasible node
    # score = dist(current, i) + alpha * min(dist(i, j) for j in feasible or depot) + beta * demand(i)
    
    # Use a simple but effective heuristic:
    # score(i) = dist(current, i) + 0.5 * dist(i, depot) + 0.3 * (demand(i) / max_demand) * avg_dist
    
    # Get distances from current to all feasible nodes
    dists_current = distance_matrix[current_node, feasible]
    
    # Get distances from each feasible node to depot
    dists_depot = distance_matrix[feasible, depot]
    
    # Normalize demands to [0, 1] for scaling
    max_demand = max(demands[n] for n in unvisited_nodes) if unvisited_nodes else 1
    if max_demand == 0:
        max_demand = 1
    
    demands_feasible = np.array([demands[n] for n in feasible], dtype=float)
    normalized_demands = demands_feasible / max_demand
    
    # Average distance scale for normalization
    avg_dist = np.mean(dists_current) if len(dists_current) > 0 else 1.0
    if avg_dist == 0:
        avg_dist = 1.0
    
    # Combined score
    alpha = 0.5   # Weight for distance to depot (encourages closing route when close)
    beta = 0.3    # Weight for demand (encourages saving capacity)
    
    scores = dists_current + alpha * dists_depot + beta * normalized_demands * avg_dist
    
    # Pick node with minimum score
    best_idx = np.argmin(scores)
    return int(feasible[best_idx])

# EVOLVE-BLOCK-END
