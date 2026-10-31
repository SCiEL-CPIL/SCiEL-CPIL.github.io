"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # Leave 5s margin for 120s limit

# Cache for pre-computed tours
_tour_cache = {}

def _get_instance_id(distance_matrix):
    """Create a hashable identifier for a distance matrix instance."""
    return hash(tuple(tuple(row) for row in distance_matrix))

def _nearest_neighbor_tour(n, dist_matrix):
    """Generate a tour using nearest neighbor heuristic."""
    visited = [0]
    unvisited = set(range(1, n))
    while unvisited:
        current = visited[-1]
        best_node = min(unvisited, key=lambda x: dist_matrix[current, x])
        visited.append(best_node)
        unvisited.remove(best_node)
    return visited

def _two_opt(tour, dist_matrix, max_iter=50):
    """Improve tour using 2-opt moves."""
    n = len(tour)
    improved = True
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j == i + 1:
                    continue
                # Calculate delta for 2-opt move
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n]
                delta = dist_matrix[a, c] + dist_matrix[b, d] - dist_matrix[a, b] - dist_matrix[c, d]
                if delta < -1e-10:
                    tour[i:j+1] = tour[i:j+1][::-1]
                    improved = True
    return tour

def _compute_best_tour(dist_matrix):
    """Compute a high-quality tour using NN + 2-Opt."""
    n = len(dist_matrix)
    tour = _nearest_neighbor_tour(n, dist_matrix)
    tour = _two_opt(tour, dist_matrix, max_iter=30)
    return tour

def _get_tour_order(dist_matrix):
    """Get or compute the cached tour order for this instance."""
    inst_id = _get_instance_id(dist_matrix)
    if inst_id not in _tour_cache:
        _tour_cache[inst_id] = _compute_best_tour(dist_matrix)
    return _tour_cache[inst_id]

def _get_next_in_tour(tour, current_node, unvisited_set):
    """Find the next node in the tour that is still unvisited."""
    n = len(tour)
    # Find current node's position in the tour
    for i in range(n):
        if tour[i] == current_node:
            # Look forward in the tour for the next unvisited node
            for j in range(1, n):
                candidate = tour[(i + j) % n]
                if candidate in unvisited_set:
                    return candidate
    # Fallback: nearest unvisited
    return min(unvisited_set, key=lambda x: dist_matrix[current_node, x])

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Check time limit
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback to nearest neighbor to ensure termination
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    if len(unvisited_nodes) == 0:
        return -1

    # Get the pre-computed tour
    tour = _get_tour_order(distance_matrix)
    unvisited_set = set(unvisited_nodes.tolist())
    
    # Find the next node in the tour that is unvisited
    next_node = _get_next_in_tour(tour, current_node, unvisited_set)
    
    return int(next_node)
# EVOLVE-BLOCK-END
