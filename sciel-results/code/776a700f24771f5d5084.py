"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = None

def _ensure_start():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME

def _time_remaining():
    start = _ensure_start()
    return 145.0 - (time.time() - start)  # safety margin of 5s before 150s


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic: Clarke-Wright savings inspired nearest neighbor with look-ahead.
    
    Strategy:
    1. If no unvisited nodes or no capacity, return depot.
    2. For each feasible unvisited node (demand fits in rest_capacity), compute a score:
       - Primary: distance from current_node to candidate
       - Secondary: savings term = dist(depot, candidate) + dist(candidate, best_next_after) - dist(depot, best_next_after)
         where best_next_after is the nearest feasible node after candidate (or depot if none)
       - Adaptive depot penalty: as rest_capacity decreases, penalize going to depot
    3. Return the node with minimum score.
    """
    _ensure_start()
    
    if not unvisited_nodes or len(unvisited_nodes) == 0:
        return int(depot)
    
    if rest_capacity <= 0:
        return int(depot)
    
    unvisited = np.array(unvisited_nodes, dtype=int)
    n = len(unvisited)
    
    # Get distances from current node to all unvisited
    dists_from_current = distance_matrix[current_node, unvisited]
    
    # Filter by capacity feasibility
    feasible_mask = demands[unvisited] <= rest_capacity
    if not np.any(feasible_mask):
        return int(depot)
    
    feasible_nodes = unvisited[feasible_mask]
    feasible_dists = dists_from_current[feasible_mask]
    feasible_demands = demands[feasible_mask]
    
    # Get distances from depot to all feasible nodes
    dists_from_depot = distance_matrix[depot, feasible_nodes]
    
    # For each feasible node, compute a look-ahead term:
    # savings = dist(depot, i) + dist(i, j) - dist(depot, j)
    # where j is the nearest other feasible node (or depot)
    # We approximate by: for each i, find the nearest j != i among feasible nodes
    # This is O(n^2) but n <= 50 so it's fine.
    
    n_f = len(feasible_nodes)
    
    # Compute pairwise distances among feasible nodes
    # dist_matrix_feasible[i, j] = distance_matrix[feasible_nodes[i], feasible_nodes[j]]
    dist_matrix_feasible = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    
    # For each feasible node i, find the nearest other feasible node j
    # If no other, use depot
    # savings_i = dist(depot, i) + dist(i, nearest_j) - dist(depot, nearest_j)
    # We want to MAXIMIZE savings (minimize negative savings)
    
    # For efficiency, compute for each i the nearest j != i
    # Set diagonal to inf
    np.fill_diagonal(dist_matrix_feasible, np.inf)
    
    nearest_j_idx = np.argmin(dist_matrix_feasible, axis=1)
    nearest_j_dist = np.min(dist_matrix_feasible, axis=1)
    
    # For nodes where no other feasible node exists (n_f == 1), nearest_j_dist is inf
    if n_f == 1:
        nearest_j_dist = np.array([distance_matrix[feasible_nodes[0], depot]])
        nearest_j_node = np.array([depot])
    else:
        nearest_j_node = feasible_nodes[nearest_j_idx]
    
    # dist(depot, nearest_j)
    dist_depot_nearest_j = distance_matrix[depot, nearest_j_node]
    
    # Savings for each feasible node i:
    # savings_i = dist(depot, i) + dist(i, nearest_j) - dist(depot, nearest_j)
    savings = dists_from_depot + nearest_j_dist - dist_depot_nearest_j
    
    # We want to minimize: dist_from_current - alpha * savings + beta * (something for depot)
    # Actually, we want to maximize savings, so we subtract alpha * savings from the distance
    
    # Adaptive depot penalty: as rest_capacity decreases relative to total remaining demand,
    # we should be more willing to go back to depot
    total_remaining_demand = np.sum(feasible_demands)
    if total_remaining_demand > 0:
        capacity_ratio = rest_capacity / total_remaining_demand
    else:
        capacity_ratio = 1.0
    
    # If capacity is tight (low ratio), we should prefer to close the route sooner
    # This means we add a penalty to staying in the route
    # Let's add a small bonus for going to depot when capacity is tight
    
    # Score for each feasible node:
    # score_i = dist_from_current[i] - alpha * savings[i]
    # where alpha controls how much we value savings
    
    alpha = 0.5  # weight for savings term
    
    scores = feasible_dists - alpha * savings
    
    # Add a small penalty for nodes that would leave very little capacity
    # If choosing node i leaves rest_capacity - demand[i] < min_demand, penalize
    min_demand = np.min(feasible_demands)
    leftover = rest_capacity - feasible_demands
    # If leftover is less than min_demand, we might be forced to go to depot next
    # This is a mild penalty
    penalty = np.where(leftover < min_demand, 0.1 * feasible_dists, 0.0)
    
    scores += penalty
    
    best_idx = np.argmin(scores)
    
    return int(feasible_nodes[best_idx])

# EVOLVE-BLOCK-END
