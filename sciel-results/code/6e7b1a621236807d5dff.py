"""CVRP Heuristic: Physics-inspired vector field with demand-weighted attraction."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_TIMEOUT = 115.0
_DEADLINE = None


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Physics-inspired heuristic: models routing as particle motion in a potential landscape.
    
    The depot emits a repulsive force (encouraging route closure when remaining capacity
    is low or when the current position is far from the depot relative to unvisited nodes).
    Unvisited customers exert attractive forces weighted by their demand and inverse
    distance, balancing load distribution and spatial proximity.
    
    The next node is selected by maximizing the net force magnitude, which dynamically
    balances between continuing the route (strong attraction to nearby high-demand nodes)
    and closing the route (strong repulsion from the depot when capacity is constrained).
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
    
    # If no feasible nodes, return depot
    if not np.any(feasible_mask):
        return int(depot)
    
    feasible_nodes = unvisited_arr[feasible_mask]
    n_feasible = len(feasible_nodes)
    
    # If at depot, pick the nearest feasible node to start the route
    if current_node == depot:
        dists = distance_matrix[current_node, feasible_nodes]
        min_idx = np.argmin(dists)
        return int(feasible_nodes[min_idx])
    
    # Extract relevant distances
    d_current_depot = distance_matrix[current_node, depot]
    d_depot_j = distance_matrix[depot, feasible_nodes]
    d_current_j = distance_matrix[current_node, feasible_nodes]
    demands_feasible = demands[feasible_nodes]
    
    # Physics-inspired scoring:
    # 
    # Attraction to customer J: proportional to demand[J] / distance(current, J)^alpha
    # This encourages visiting high-demand nearby nodes.
    # 
    # Repulsion from depot: proportional to 1 / distance(current, depot)^beta
    # This encourages closing the route when the current position is far from the depot
    # relative to the remaining capacity.
    # 
    # We combine these into a net force score for each candidate J.
    
    # Attraction score: demand-weighted inverse distance
    # Use a soft minimum distance to avoid division by zero
    min_dist = 1e-6
    d_current_j_safe = np.maximum(d_current_j, min_dist)
    
    # alpha controls how strongly distance affects attraction (higher = more local)
    alpha = 1.2
    attraction = demands_feasible / (d_current_j_safe ** alpha)
    
    # Repulsion from depot: stronger when current node is far from depot
    # and when remaining capacity is low (suggesting route should close soon)
    d_current_depot_safe = max(d_current_depot, min_dist)
    
    # Capacity pressure: higher when rest_capacity is low relative to total remaining demand
    total_remaining_demand = np.sum(demands_feasible)
    if total_remaining_demand > 0:
        capacity_pressure = 1.0 - (rest_capacity / max(total_remaining_demand, 1.0))
    else:
        capacity_pressure = 0.0
    
    # Clamp capacity pressure to [0, 1]
    capacity_pressure = np.clip(capacity_pressure, 0.0, 1.0)
    
    # Repulsion strength: increases with distance from depot and capacity pressure
    # beta controls how strongly distance affects repulsion
    beta = 0.8
    repulsion_base = 1.0 / (d_current_depot_safe ** beta)
    
    # Scale repulsion by capacity pressure: more pressure = stronger pull to depot
    # The repulsion score for closing the route is the same for all J, so we
    # add it as a constant offset that makes closing more attractive when
    # capacity pressure is high.
    repulsion_score = repulsion_base * capacity_pressure * 10.0
    
    # Net score for each candidate J:
    # score[J] = attraction[J] - repulsion_score + savings_bonus[J]
    # 
    # We also include a savings component to maintain some of the classic
    # savings heuristic behavior: S(C, J) = d(C, Depot) + d(Depot, J) - d(C, J)
    savings = d_current_depot + d_depot_j - d_current_j
    
    # Normalize savings to be comparable in scale to attraction
    # Use a simple normalization: divide by a characteristic distance
    max_savings = np.max(savings)
    if max_savings > 0:
        savings_normalized = savings / max_savings
    else:
        savings_normalized = np.zeros_like(savings)
    
    # Combine: attraction + weighted savings - repulsion
    # gamma weights the savings component
    gamma = 0.5
    total_score = attraction + gamma * savings_normalized - repulsion_score
    
    # If the best total score is significantly negative, it's better to close the route
    # Use a threshold that accounts for the repulsion
    if np.max(total_score) < -repulsion_score * 0.5:
        return int(depot)
    
    max_idx = np.argmax(total_score)
    
    return int(feasible_nodes[max_idx])
# EVOLVE-BLOCK-END
