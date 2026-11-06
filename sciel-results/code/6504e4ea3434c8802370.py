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
    
    # Filter by capacity
    # demands is likely a list or array indexed by node id
    # We need to check demands[node] <= rest_capacity
    feasible_mask = np.array([demands[node] <= rest_capacity for node in unvisited], dtype=bool)
    
    if not np.any(feasible_mask):
        return depot
        
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    # Calculate scores using a refined savings-based heuristic
    # The savings of adding j to the current route (ending at i) vs. serving j separately:
    #   savings(j) = d(depot, j) + d(j, next_after_j) - d(i, j)
    # But since we don't know next_after_j yet, we approximate.
    #
    # A common and effective heuristic is to minimize:
    #   d(i, j) - alpha * d(depot, j)
    # where alpha is a tuning parameter. alpha=1 is the classic savings.
    #
    # Let's try a soft approach: use a temperature to create a smooth surrogate.
    # score(j) = d(i, j) - T * log(1 + d(depot, j))
    # This gives more weight to nodes far from depot when T is large.
    
    # d(i, j)
    dists_from_current = distance_matrix[current_node, feasible_nodes]
    # d(depot, j)
    dists_from_depot = distance_matrix[depot, feasible_nodes]
    
    # Classic savings: minimize d(i,j) - d(depot,j)
    # But let's add a small penalty for nodes that are close to depot (they're "cheap" to serve alone)
    # and reward nodes far from depot (they "save" more by being included).
    
    # Let's try: score = d(i,j) - d(depot,j) + beta * d(i,j) * d(depot,j) / (d(i,j) + d(depot,j) + 1e-9)
    # The last term is a harmonic-mean-like penalty that is small when either distance is small.
    
    # Actually, let's stick with the classic savings but add a small random perturbation
    # to break ties and explore different solutions. However, since this is a deterministic
    # evaluation, let's just use the classic savings with a slight modification.
    
    # Modified savings: score(j) = d(i,j) - d(depot,j) + gamma * (1 / (d(i,j) + 1))
    # The gamma term slightly penalizes very close nodes to encourage visiting farther nodes
    # when the savings are similar.
    
    gamma = 0.1
    scores = dists_from_current - dists_from_depot + gamma / (dists_from_current + 1.0)
    
    # Pick the one with minimum score
    min_idx = np.argmin(scores)
    next_node = feasible_nodes[min_idx]
    
    return int(next_node)
# EVOLVE-BLOCK-END
