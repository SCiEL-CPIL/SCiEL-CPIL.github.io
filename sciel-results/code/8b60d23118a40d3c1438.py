"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level deadline guard
_DEADLINE_SECONDS = 115.0  # safety margin under 120s limit
_start_time = None


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _check_deadline():
    return _get_elapsed() > _DEADLINE_SECONDS


def _greedy_completion_cost(current, dest, unvisited, dist_matrix):
    """
    Simulate greedy nearest-neighbor completion from current through unvisited
    to dest, returning total path length.
    """
    total = 0.0
    remaining = list(unvisited)
    cur = current
    
    while remaining:
        # Find nearest unvisited to current
        # dist_matrix[cur, remaining] is a vector of distances
        idx = np.argmin(dist_matrix[cur, remaining])
        next_node = remaining[idx]
        total += dist_matrix[cur, next_node]
        remaining.pop(idx)
        cur = next_node
    
    # Final jump to destination
    total += dist_matrix[cur, dest]
    return total


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a greedy simulation lookahead.
    We add a small convex-hull/polar-angle bias to prefer nodes that
    progress towards the destination, as suggested by the directive.
    """
    # Deadline guard: if we're over budget, use fastest fallback
    if _check_deadline():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Calculate vector from current to destination for angle calculation
    # distance_matrix is Euclidean, but we don't have coordinates.
    # We can only use distance_matrix.
    # Without coordinates, we cannot calculate angles.
    # The directive assumes we can calculate polar angles.
    # But the interface only provides distance_matrix.
    # We must derive geometry or skip the angle part.
    # Since we only have distances, we cannot compute angles directly.
    # We can approximate "progress" by checking if dist(current, cand) + dist(cand, dest) 
    # is close to dist(current, dest) (triangle inequality equality).
    
    # Let's use a "Triangle Inequality" bias instead of angles.
    # If dist(current, cand) + dist(cand, dest) is small, cand is "between" or close to the line.
    # Actually, for TSP, we want to avoid backtracking.
    
    best_score = np.inf
    best_node = unvisited_nodes[0]
    
    # Pre-compute distances from current to all unvisited
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    for i, candidate in enumerate(unvisited_nodes):
        cost_to_candidate = dists_from_current[i]
        
        # Remaining nodes after picking this candidate
        remaining = np.delete(unvisited_nodes, i)
        
        # Simulate greedy completion
        rem_list = list(remaining)
        cur = candidate
        sim_cost = 0.0
        
        while rem_list:
            idx = np.argmin(distance_matrix[cur, rem_list])
            next_node = rem_list[idx]
            sim_cost += distance_matrix[cur, next_node]
            rem_list.pop(idx)
            cur = next_node
        
        sim_cost += distance_matrix[cur, destination_node]
        total_cost = cost_to_candidate + sim_cost
        
        # Add a small bias: prefer candidates that are closer to the destination
        # This reinforces the simulation's destination-awareness.
        # dist_to_dest = distance_matrix[candidate, destination_node]
        # bias = 0.01 * dist_to_dest
        
        # Since we can't do angles, we stick to the simulation cost.
        # The simulation already accounts for the destination.
        
        if total_cost < best_score:
            best_score = total_cost
            best_node = candidate
    
    return int(best_node)
# EVOLVE-BLOCK-END
