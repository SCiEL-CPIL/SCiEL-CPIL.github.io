"""Capacitated VRP heuristic: Route-level look-ahead with Nearest Neighbor completion."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 115.0  # Safety margin before 150s limit
_DEADLINE = _start_time + _TIMEOUT


def _complete_route_nn(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Complete a route starting from current_node using Nearest Neighbor heuristic.
    Returns the total cost of the completed route (including return to depot).
    """
    if not unvisited_nodes:
        return distance_matrix[current_node, depot]
    
    demands_arr = np.asarray(demands)
    remaining = set(unvisited_nodes)
    curr = current_node
    cap = rest_capacity
    cost = 0.0
    
    while remaining:
        # Find feasible nodes
        feasible = [n for n in remaining if demands_arr[n] <= cap]
        if not feasible:
            break
        
        # Pick nearest feasible node
        feasible_arr = np.array(feasible, dtype=int)
        dists = distance_matrix[curr, feasible_arr]
        idx = np.argmin(dists)
        next_node = feasible_arr[idx]
        
        cost += distance_matrix[curr, next_node]
        curr = next_node
        cap -= demands_arr[curr]
        remaining.remove(curr)
        
        if time.time() > _DEADLINE:
            break
    
    cost += distance_matrix[curr, depot]
    return cost


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a route-level look-ahead strategy.
    For each feasible candidate, simulate the rest of the route using NN completion.
    Returns 0 (depot) if no feasible node should be visited.
    """
    if time.time() > _DEADLINE:
        return depot

    if len(unvisited_nodes) == 0:
        return depot

    unvisited = np.array(unvisited_nodes, dtype=int)
    
    # Ensure demands is indexable
    if isinstance(demands, list):
        demands_arr = np.array(demands)
    else:
        demands_arr = np.asarray(demands)

    # Filter by capacity
    feasible_mask = demands_arr[unvisited] <= rest_capacity
    feasible_nodes = unvisited[feasible_mask]

    if len(feasible_nodes) == 0:
        return depot

    # If at depot, pick the nearest feasible node (standard)
    if current_node == depot:
        dists = distance_matrix[current_node, feasible_nodes]
        min_idx = np.argmin(dists)
        return int(feasible_nodes[min_idx])

    # For each feasible candidate, estimate the total route cost
    # if we visit this node next and then complete the route with NN.
    
    best_cost = float('inf')
    best_node = int(feasible_nodes[0])
    
    for node in feasible_nodes:
        if time.time() > _DEADLINE:
            break
            
        # Simulate: visit 'node' next, then complete with NN
        remaining_after = set(unvisited_nodes)
        remaining_after.discard(node)
        
        # Cost to reach this node
        cost_to_node = distance_matrix[current_node, node]
        
        # Complete the rest of the route from 'node' using NN
        # Note: We pass the state after visiting 'node'
        rest_cost = _complete_route_nn(
            node, depot, remaining_after, 
            rest_capacity - demands_arr[node], 
            demands, distance_matrix
        )
        
        total_cost = cost_to_node + rest_cost
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_node = int(node)
    
    return best_node
# EVOLVE-BLOCK-END
