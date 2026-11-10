"""CVRP heuristic: Physics-inspired force field with savings and look-ahead refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = None

# Hyperparameters
W_SAVINGS = 0.5
W_FORCE = 0.3
W_LOOKAHEAD = 0.2
CLOSING_THRESHOLD = 0.0


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a hybrid physics-inspired force field and savings heuristic.
    
    The approach combines:
    1. A force field model where the depot repels and customers attract (weighted by demand)
    2. Classic savings heuristic for route efficiency
    3. Look-ahead penalty for exit cost
    
    Returns 0 (depot) if no feasible node can be added or if the time limit is reached.
    """
    global _start_time, _DEADLINE
    
    if _start_time is None:
        _start_time = time.time()
        _DEADLINE = _start_time + _TIMEOUT
    
    # Deadline guard
    if time.time() > _DEADLINE:
        return int(depot)
    
    # Convert unvisited_nodes to numpy array
    if isinstance(unvisited_nodes, (list, tuple)):
        unvisited_arr = np.array(unvisited_nodes, dtype=int)
    else:
        unvisited_arr = np.asarray(unvisited_nodes, dtype=int)
    
    # If no unvisited nodes, return depot
    if len(unvisited_arr) == 0:
        return int(depot)
    
    # Find feasible nodes
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    
    if not np.any(feasible_mask):
        return int(depot)
    
    feasible_nodes = unvisited_arr[feasible_mask]
    n_feasible = len(feasible_nodes)
    
    # If at depot, pick the node with highest "potential" (far from depot, well-connected)
    if current_node == depot:
        d_depot = distance_matrix[depot, feasible_nodes]
        demands_f = demands[feasible_nodes]
        
        # Compute average distance from each feasible node to other unvisited nodes
        if n_feasible > 1:
            avg_conn = np.array([
                np.mean(distance_matrix[f, unvisited_arr]) for f in feasible_nodes
            ])
            max_conn = np.max(avg_conn) if np.max(avg_conn) > 0 else 1.0
            norm_conn = avg_conn / max_conn
        else:
            norm_conn = np.zeros(n_feasible)
        
        # Score: prefer far from depot (high savings potential) and well-connected
        max_depot = np.max(d_depot) if np.max(d_depot) > 0 else 1.0
        norm_depot = d_depot / max_depot
        norm_demand = demands_f / max(1, np.max(demands_f))
        
        scores = 0.5 * norm_depot + 0.3 * norm_demand - 0.2 * norm_conn
        best_idx = np.argmax(scores)
        return int(feasible_nodes[best_idx])
    
    # Not at depot: compute hybrid score
    d_current_depot = distance_matrix[current_node, depot]
    d_depot_j = distance_matrix[depot, feasible_nodes]
    d_current_j = distance_matrix[current_node, feasible_nodes]
    demands_j = demands[feasible_nodes]
    
    # 1. Classic Savings
    savings = d_current_depot + d_depot_j - d_current_j
    
    # 2. Force Field Model
    # The depot exerts a repulsive force (pushes away from depot)
    # Customers exert attractive forces weighted by demand and inverse distance
    # Net force on candidate j: attraction from other customers - repulsion from depot
    
    if n_feasible > 1:
        # Distance matrix among feasible nodes
        d_j_k = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
        np.fill_diagonal(d_j_k, np.inf)
        
        # Attraction from other customers: sum of (demand_k / dist(j,k)) for k != j
        # This represents "pull" from other customers toward j
        attraction = np.zeros(n_feasible)
        for i in range(n_feasible):
            mask = np.ones(n_feasible, dtype=bool)
            mask[i] = False
            if np.any(mask):
                # Weighted sum: demand_k / dist(j,k)
                attraction[i] = np.sum(demands_j[mask] / d_j_k[i, mask])
        
        # Repulsion from depot: proportional to 1 / dist(j, depot)
        # But we want to be pulled toward depot when close, so repulsion is actually
        # a penalty for being far from depot in terms of "wasted" capacity
        # Better: repulsion from depot means we want to minimize distance to depot
        # So force = -dist(j, depot) (we want to go back to depot)
        # But that conflicts with savings. Let's use: force_repel = -d_depot_j
        # Actually, the physics model: depot is a sink, customers are sources
        # Net force on j = sum_k (demand_k * unit_vector(j->k) / dist(j,k)^2) - repulsion_from_depot
        
        # Simplified: use attraction as a bonus (higher is better)
        max_attraction = np.max(attraction) if np.max(attraction) > 0 else 1.0
        norm_attraction = attraction / max_attraction
    else:
        norm_attraction = np.zeros(n_feasible)
    
    # 3. Look-Ahead Penalty
    if n_feasible > 1:
        d_j_k = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
        np.fill_diagonal(d_j_k, np.inf)
        min_dist_to_next = np.min(d_j_k, axis=1)
    else:
        min_dist_to_next = d_depot_j.copy()
    
    # Normalize
    max_savings = np.max(savings) if np.max(savings) > 0 else 1.0
    norm_savings = np.clip(savings / max_savings, 0, 1)
    max_lookahead = np.max(min_dist_to_next) if np.max(min_dist_to_next) > 0 else 1.0
    norm_lookahead = min_dist_to_next / max_lookahead
    
    # Final Score
    scores = (W_SAVINGS * norm_savings + 
              W_FORCE * norm_attraction - 
              W_LOOKAHEAD * norm_lookahead)
    
    # Decision: If the best score is not significantly positive, close the route
    if np.max(scores) <= CLOSING_THRESHOLD:
        return int(depot)
    
    best_idx = np.argmax(scores)
    best_node = feasible_nodes[best_idx]
    
    # Route closing decision: check if visiting best_node is worth it
    cost_visit = d_current_j[best_idx] + d_depot_j[best_idx]
    cost_return = d_current_depot
    
    # If visiting costs more than 1.15x returning, close the route
    if cost_visit > cost_return * 1.15:
        return int(depot)
    
    return int(best_node)
# EVOLVE-BLOCK-END
