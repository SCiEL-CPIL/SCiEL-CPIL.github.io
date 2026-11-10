"""CVRP heuristic: Physics-inspired vector field with demand-weighted attraction and capacity-dependent depot repulsion."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = None


def _physics_score(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix, total_capacity):
    """
    Compute a physics-inspired score for each feasible candidate node.
    
    The model treats routing as particle motion in a potential landscape:
    - Unvisited customers exert attractive forces weighted by demand and inverse distance
    - The depot exerts a repulsive force that increases as capacity is consumed
    - The next node is the one that maximizes net attractive force while avoiding depot repulsion
    
    Returns the best candidate node or depot if closing is optimal.
    """
    n = len(distance_matrix)
    
    # Precompute distances from current node to all nodes
    dist_from_current = distance_matrix[current_node, :]
    dist_from_depot = distance_matrix[depot, :]
    
    # Capacity pressure: 0 when full capacity, 1 when no capacity left
    capacity_pressure = 1.0 - (rest_capacity / total_capacity) if total_capacity > 0 else 1.0
    
    # Depot repulsion strength increases with capacity pressure
    # When capacity is tight, we should prefer to close the route sooner
    depot_repulsion_weight = 0.5 + 1.5 * capacity_pressure
    
    best_candidate = None
    best_score = float('-inf')
    
    # Cost of closing now
    close_cost = dist_from_current[depot]
    
    for candidate in unvisited_nodes:
        if demands[candidate] > rest_capacity:
            continue
        
        d_cur_cand = dist_from_current[candidate]
        d_depot_cand = dist_from_depot[candidate]
        
        # Attractive force from candidate: inversely proportional to distance,
        # weighted by demand (larger demand = more "mass" to attract)
        # Use a softened inverse to avoid division by zero
        attract_weight = demands[candidate] / (1.0 + d_cur_cand)
        
        # Repulsive force from depot: increases with capacity pressure
        # The depot "pushes back" harder when we're close to capacity limit
        repel_weight = depot_repulsion_weight / (1.0 + d_depot_cand)
        
        # Net force: attraction minus repulsion
        # We want to maximize this, but also consider the actual routing cost
        net_force = attract_weight - repel_weight
        
        # Combine with a routing cost consideration:
        # The savings from visiting candidate vs closing now
        savings = close_cost + d_depot_cand - d_cur_cand
        
        # Final score: balance of physics force and routing savings
        # Normalize savings to be comparable to force
        score = net_force + 0.3 * savings / (1.0 + max(close_cost, 1e-10))
        
        if score > best_score:
            best_score = score
            best_candidate = candidate
    
    # Decision: if no feasible candidate or best score is negative, close
    if best_candidate is None or best_score < 0:
        return int(depot)
    
    return int(best_candidate)


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a physics-inspired vector field heuristic.
    
    The model treats routing as particle motion where:
    - Unvisited customers exert attractive forces weighted by demand
    - The depot exerts repulsive force that increases with capacity consumption
    - The next node maximizes the net force magnitude
    
    Returns 0 (depot) if no feasible node can be added or if closing is better.
    """
    global _start_time, _DEADLINE
    
    if _start_time is None:
        _start_time = time.time()
        _DEADLINE = _start_time + _TIMEOUT
    
    # Deadline guard
    if time.time() > _DEADLINE:
        return int(depot)
    
    # Convert unvisited_nodes to list if needed
    if isinstance(unvisited_nodes, (list, tuple)):
        unvisited_list = list(unvisited_nodes)
    else:
        unvisited_list = list(unvisited_nodes)
    
    # If no unvisited nodes, return depot
    if len(unvisited_list) == 0:
        return int(depot)
    
    # Find feasible nodes
    feasible = [n for n in unvisited_list if demands[n] <= rest_capacity]
    
    if not feasible:
        return int(depot)
    
    # If at depot, pick the nearest feasible node to start the route
    if current_node == depot:
        best_node = min(feasible, key=lambda n: distance_matrix[current_node, n])
        return int(best_node)
    
    # Compute total capacity for normalization
    total_capacity = max(1, int(np.sum(demands)))
    
    # Use physics-inspired scoring
    return _physics_score(current_node, depot, feasible, rest_capacity, demands, distance_matrix, total_capacity)
# EVOLVE-BLOCK-END
