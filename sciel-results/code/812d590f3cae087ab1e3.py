"""Capacitated VRP heuristic: Clarke-Wright savings + nearest-neighbor look-ahead + adaptive depot penalty."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer or return depot (0) to close a route.
    
    Strategy:
    1. If no unvisited nodes or capacity too low for any, return depot.
    2. Score each candidate by:
       - Clarke-Wright savings term: reward chaining (savings of going via this node vs two separate trips)
       - Nearest-neighbor look-ahead: prefer nodes that lead to dense clusters
       - Adaptive depot penalty: as capacity empties, penalize long jumps to depot
    3. Pick the best-scoring feasible node.
    """
    n = len(unvisited_nodes)
    if n == 0:
        return depot
    
    # Convert to numpy arrays for speed
    unvisited = np.array(unvisited_nodes, dtype=np.int64)
    dist = np.asarray(distance_matrix)
    dem = np.asarray(demands)
    
    # Filter feasible nodes (demand fits in remaining capacity)
    feasible_mask = dem[unvisited] <= rest_capacity
    feasible = unvisited[feasible_mask]
    
    if len(feasible) == 0:
        return depot
    
    # If only one feasible, take it
    if len(feasible) == 1:
        return int(feasible[0])
    
    # --- Compute scores for each feasible candidate ---
    # Distance from current node to each candidate
    dist_from_current = dist[current_node, feasible]
    
    # Clarke-Wright savings: 
    # savings(i) = d(current, i) + d(i, depot) - d(current, depot)
    # Actually the standard savings is: d(depot, i) + d(i, j) - d(depot, j) for merging
    # Here we use a look-ahead: how much do we save by going to i now vs going to depot?
    # A better proxy: prefer nodes that are close to current AND close to depot (good endpoints)
    # savings_proxy = d(current, i) + d(i, depot) - d(current, depot)
    # Lower is better (less "penalty")
    d_current_depot = dist[current_node, depot]
    d_i_depot = dist[feasible, depot]
    savings_proxy = dist_from_current + d_i_depot - d_current_depot
    
    # Nearest-neighbor look-ahead: for each candidate i, compute the min distance from i to any other feasible node
    # This rewards chaining into dense clusters
    # dist from each feasible node to all feasible nodes
    dist_matrix_feasible = dist[np.ix_(feasible, feasible)]
    # Set diagonal to inf so we don't pick self
    np.fill_diagonal(dist_matrix_feasible, np.inf)
    min_dist_to_other = np.min(dist_matrix_feasible, axis=1)
    
    # Adaptive depot penalty: as rest_capacity decreases, penalize nodes far from depot
    # This encourages closing routes when capacity is low
    cap_ratio = rest_capacity / 50.0  # assuming capacity 50
    # Scale penalty: when cap_ratio is low, strongly prefer nodes close to depot
    depot_proximity_penalty = (1.0 - cap_ratio) * d_i_depot
    
    # Combine scores: we want to minimize a composite cost
    # - savings_proxy: lower is better (prefer nodes that don't add much distance)
    # - min_dist_to_other: lower is better (prefer nodes with nearby neighbors for chaining)
    # - depot_proximity_penalty: lower is better (prefer nodes close to depot when capacity is low)
    
    # Normalize each term to [0, 1] for fair combination
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    norm_savings = normalize(savings_proxy)
    norm_lookahead = normalize(min_dist_to_other)
    norm_depot_pen = normalize(depot_proximity_penalty)
    
    # Weights: savings is strongest, then look-ahead, then depot penalty
    w_savings = 0.5
    w_lookahead = 0.3
    w_depot = 0.2
    
    # Composite score (lower is better)
    score = w_savings * norm_savings + w_lookahead * norm_lookahead + w_depot * norm_depot_pen
    
    # Pick the best
    best_idx = np.argmin(score)
    return int(feasible[best_idx])
# EVOLVE-BLOCK-END
