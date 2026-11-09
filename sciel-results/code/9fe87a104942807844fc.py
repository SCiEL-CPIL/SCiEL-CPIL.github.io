"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global cache for savings-based heuristics
_savings_cache = {}


def _compute_savings(depot, demands, distance_matrix):
    """Compute Clarke-Wright savings for all pairs of customers."""
    n = len(demands)
    # savings[i, j] = d(depot, i) + d(depot, j) - d(i, j)
    savings = np.zeros((n, n), dtype=np.float64)
    depot_d = distance_matrix[depot]
    for i in range(1, n):
        for j in range(i + 1, n):
            s = depot_d[i] + depot_d[j] - distance_matrix[i, j]
            savings[i, j] = s
            savings[j, i] = s
    return savings


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Select the next node using a Clarke-Wright savings-inspired heuristic.
    
    Strategy: Among feasible customers (demand fits in rest_capacity),
    score each candidate by:
      score = savings(current_node, candidate) - alpha * distance(current_node, candidate)
    where savings is the Clarke-Wright saving of merging current and candidate.
    
    This favors customers that would save the most if visited now vs. later,
    while penalizing very distant nodes.
    """
    if not unvisited_nodes or len(unvisited_nodes) == 0:
        return depot
    
    # Check if any feasible customers exist
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    
    if not feasible:
        return depot
    
    # If only one feasible, take it
    if len(feasible) == 1:
        return int(feasible[0])
    
    # Use cached savings if available, else compute
    cache_key = (depot, len(demands))
    if cache_key not in _savings_cache:
        _savings_cache[cache_key] = _compute_savings(depot, demands, distance_matrix)
    savings = _savings_cache[cache_key]
    
    # Score each feasible candidate
    best_node = None
    best_score = -np.inf
    
    dist_from_current = distance_matrix[current_node]
    
    for node in feasible:
        # Clarke-Wright saving: how much we save by visiting node after current
        # instead of going depot->current->depot and depot->node->depot separately
        # savings[current_node, node] = d(depot, current) + d(depot, node) - d(current, node)
        s = savings[current_node, node]
        
        # Distance penalty
        d = dist_from_current[node]
        
        # Combined score: high savings and low distance are good
        # Normalize: use savings as primary, distance as secondary
        score = s - 0.3 * d
        
        if score > best_score:
            best_score = score
            best_node = node
    
    return int(best_node)
# EVOLVE-BLOCK-END
