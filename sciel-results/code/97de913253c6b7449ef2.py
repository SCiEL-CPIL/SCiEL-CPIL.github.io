import numpy as np
import time

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Convert to numpy arrays if necessary for vectorized operations
    if len(unvisited_nodes) == 0:
        return 0
    
    # If we are at the depot and there's no capacity or no feasible nodes, return depot
    # But typically we start at depot with full capacity
    
    unvisited = np.array(unvisited_nodes)
    
    # Filter for feasible nodes based on capacity
    # demands is likely an array or list indexed by node id
    # rest_capacity is the remaining capacity
    feasible_mask = np.array([demands[node] <= rest_capacity for node in unvisited])
    
    if not np.any(feasible_mask):
        return 0
    
    feasible_nodes = unvisited[feasible_mask]
    
    # Calculate scores for feasible nodes
    # We want to minimize the total route length.
    # A good heuristic is the "savings" method from Clarke-Wright, adapted for online decision making.
    # Savings of visiting node i after current j instead of closing route and starting new one:
    # Savings = dist(depot, j) + dist(depot, i) - dist(j, i)
    # We want to MAXIMIZE savings, which is equivalent to MINIMIZING -savings.
    # -savings = dist(j, i) - dist(depot, j) - dist(depot, i)
    
    # However, we also need to consider that if we go to i, we might not be able to visit other nodes later.
    # A simple greedy approach is to pick the node that maximizes savings.
    
    dist_current = distance_matrix[current_node, feasible_nodes]
    dist_depot_to_current = distance_matrix[depot, current_node]
    dist_depot_to_nodes = distance_matrix[depot, feasible_nodes]
    
    # savings = dist_depot_to_current + dist_depot_to_nodes - dist_current
    # We want to maximize savings.
    # Let's compute the negative savings to minimize.
    negative_savings = dist_current - dist_depot_to_current - dist_depot_to_nodes
    
    # Pick the node with the minimum negative savings (i.e., maximum savings)
    best_idx = np.argmin(negative_savings)
    best_node = feasible_nodes[best_idx]
    
    return int(best_node)
