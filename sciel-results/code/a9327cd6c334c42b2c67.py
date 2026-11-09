"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache to avoid recomputing expensive operations across calls within the same tour
_tour_start_time = None
_instance_cache = {}

def _get_instance_key(distance_matrix):
    """Create a lightweight key for the distance matrix to cache instance-specific computations."""
    # Use the shape and a hash of the first few elements to identify the instance
    # This is a heuristic; collisions are unlikely for different instances
    shape = distance_matrix.shape
    # Take a small sample of the matrix to create a hash
    sample = distance_matrix[::10, ::10].flatten()
    if len(sample) > 100:
        sample = sample[:100]
    return (shape, tuple(sample))

def _compute_angle_indexing(distance_matrix, n):
    """Precompute angles from centroid for each node to help with farthest-point heuristic."""
    # Compute centroid
    # We don't have coordinates, only distances. So we can't directly compute angles.
    # Instead, we'll use a different approach: precompute pairwise distances sorted by distance
    # for each node, which helps in nearest-neighbor-like heuristics.
    
    # For a constructive heuristic with only distance matrix, we can precompute:
    # 1. For each node, the sorted list of other nodes by distance
    # 2. This allows faster nearest-neighbor selection
    
    sorted_neighbors = {}
    for i in range(n):
        dists = distance_matrix[i]
        # Get indices sorted by distance
        sorted_idx = np.argsort(dists)
        sorted_neighbors[i] = sorted_idx
    
    return sorted_neighbors

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _tour_start_time, _instance_cache
    
    # Deadline guard: track elapsed time
    if _tour_start_time is None:
        _tour_start_time = time.time()
    
    elapsed = time.time() - _tour_start_time
    # Safety margin: return simple nearest neighbor if we're close to timeout
    if elapsed > 115:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Check if we have cached instance-specific data
    key = _get_instance_key(distance_matrix)
    n = distance_matrix.shape[0]
    
    if key not in _instance_cache:
        # Compute sorted neighbors for this instance
        sorted_neighbors = _compute_angle_indexing(distance_matrix, n)
        _instance_cache[key] = sorted_neighbors
    else:
        sorted_neighbors = _instance_cache[key]
    
    # Strategy: Use a combination of nearest neighbor and farthest-point heuristics
    # with a bias toward nodes that are far from the current path but close to destination
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    if len(unvisited_nodes) == 2:
        a, b = unvisited_nodes
        # Choose the one that creates a shorter path to destination
        da = distance_matrix[a, destination_node]
        db = distance_matrix[b, destination_node]
        # Also consider the cost of going from current to the chosen node
        ca = distance_matrix[current_node, a] + da
        cb = distance_matrix[current_node, b] + db
        if ca < cb:
            return int(a)
        else:
            return int(b)
    
    # Phase-aware heuristic: blend nearest-neighbor and destination-proximity
    # based on how many nodes remain unvisited.
    # Early in the tour: pure NN (alpha=1, beta=0)
    # Late in the tour: increase beta to pull toward destination
    # This avoids the long closing edge while keeping early construction efficient.
    
    n_total = n  # total nodes in instance
    n_unvisited = len(unvisited_nodes)
    # Progress: 0 at start (many unvisited), 1 at end (few unvisited)
    # Use a smooth transition: progress increases as we get closer to finishing
    progress = 1.0 - (n_unvisited / max(n_total, 1))
    
    # Phase-aware weights:
    # First ~40% of picks (progress < 0.4): pure NN
    # Last ~10% of picks (progress > 0.9): strong destination bias
    # Middle: smooth blend
    if progress < 0.4:
        alpha = 1.0
        beta = 0.0
    elif progress > 0.9:
        alpha = 0.5
        beta = 0.5
    else:
        # Smooth transition from pure NN to balanced
        t = (progress - 0.4) / 0.5  # 0 to 1 in this range
        alpha = 1.0 - 0.5 * t
        beta = 0.5 * t
    
    # Compute distances from current node to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Compute distances from destination to all unvisited
    dist_to_destination = distance_matrix[destination_node, unvisited_nodes]
    
    # Weighted score: lower is better
    scores = alpha * dist_from_current + beta * dist_to_destination
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# Reset cache when a new tour starts (detected by destination_node being 0 or similar)
# This is a heuristic; in practice, the evaluator likely calls this for each step of a tour
# We can reset the cache when we detect a new tour, but it's tricky without explicit signaling.
# For now, we rely on the key-based caching which should handle different instances correctly.
# EVOLVE-BLOCK-END
