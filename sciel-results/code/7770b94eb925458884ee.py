"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline guard
_START_TIME = None
_DEADLINE = 115.0  # 5s margin before 120s

def _init_deadline():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _DEADLINE

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    _init_deadline()
    
    if len(unvisited_nodes) == 0:
        return depot

    # Convert to numpy array for fast indexing
    unvisited = np.array(unvisited_nodes, dtype=np.int64)
    n_unvisited = len(unvisited)
    
    # Filter by capacity
    feasible_mask = np.array([demands[node] <= rest_capacity for node in unvisited], dtype=bool)
    
    if not np.any(feasible_mask):
        return depot
        
    feasible_nodes = unvisited[feasible_mask]
    n_feasible = len(feasible_nodes)
    
    if n_feasible == 0:
        return depot

    # If only one feasible node, go there
    if n_feasible == 1:
        return int(feasible_nodes[0])

    # Get distances
    d_current = distance_matrix[current_node, feasible_nodes]
    d_depot = distance_matrix[depot, feasible_nodes]
    
    # Savings score: d(current, j) - d(depot, j)
    # Lower means more savings (or less penalty)
    savings = d_current - d_depot
    
    # Additional heuristic: prefer nodes that are close to the depot
    # This helps in closing routes efficiently
    # Weight: 0.7 * savings + 0.3 * distance to depot
    # Actually, let's think about this differently.
    
    # The savings heuristic says: merging two routes saves d(depot, i) + d(depot, j) - d(i, j)
    # Here we are adding j to a route ending at i.
    # The "cost" of adding j to the current route is d(i, j) + d(j, depot) [if we close after]
    # The "cost" of starting a new route for j is d(depot, j) + d(j, depot) = 2*d(depot, j)
    # So the difference is d(i, j) + d(j, depot) - 2*d(depot, j) = d(i, j) - d(depot, j)
    # So minimizing d(i, j) - d(depot, j) is correct for maximizing savings.
    
    # Let's try a more sophisticated score that also considers the future.
    # A node that is close to many other unvisited nodes might be better visited later
    # to allow chaining. But this is complex.
    
    # Let's stick with the savings but add a small bias towards closer nodes to the depot
    # to encourage closing the route sooner if savings are similar.
    
    # Normalized scores
    s_min = np.min(savings)
    s_max = np.max(savings)
    if s_max > s_min:
        s_norm = (savings - s_min) / (s_max - s_min)
    else:
        s_norm = np.zeros_like(savings)
    
    d_min = np.min(d_depot)
    d_max = np.max(d_depot)
    if d_max > d_min:
        d_norm = (d_depot - d_min) / (d_max - d_min)
    else:
        d_norm = np.zeros_like(d_depot)
    
    # Combine: primarily savings, secondarily distance to depot
    # We want to minimize savings, and also minimize distance to depot
    # So score = w1 * s_norm + w2 * d_norm
    # Lower is better
    w1 = 0.8
    w2 = 0.2
    
    scores = w1 * s_norm + w2 * d_norm
    
    # Pick the one with minimum score
    min_idx = np.argmin(scores)
    next_node = feasible_nodes[min_idx]
    
    return int(next_node)
# EVOLVE-BLOCK-END
