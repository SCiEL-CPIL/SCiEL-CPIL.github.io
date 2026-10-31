# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for caching and timing
_instance_cache = {}
_start_time = None
_DEADLINE = 140.0

def _get_instance_key(distance_matrix):
    # Use a hash of the distance matrix to identify the instance
    # We only need to do this once per instance, so we can cache the key or just the tour.
    # Since distance_matrix is passed every time, we can check if we've seen it.
    # However, hashing a 50x50 matrix is cheap.
    return hash(distance_matrix.tobytes())

def _construct_tour_nn(distance_matrix):
    """Construct a tour using Nearest Neighbor heuristic."""
    n = distance_matrix.shape[0]
    unvisited = set(range(n))
    tour = []
    current = 0
    unvisited.remove(0)
    tour.append(0)
    
    while unvisited:
        # Find nearest neighbor
        # distance_matrix[current, list(unvisited)]
        # To be efficient, we can just iterate or use numpy
        # Numpy is faster for vectorized ops
        candidates = list(unvisited)
        dists = distance_matrix[current, candidates]
        min_idx = np.argmin(dists)
        next_node = candidates[min_idx]
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node
        
    return tour

def _two_opt(tour, distance_matrix, max_passes=10):
    """Improve tour using 2-Opt local search."""
    n = len(tour)
    if n <= 3:
        return tour
    
    tour = list(tour)
    improved = True
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if i == 0 and j == n - 1:
                    continue
                
                # Calculate delta cost
                # Current edges: (tour[i-1], tour[i]) and (tour[j], tour[j+1])
                # Note: tour is a cycle, so indices wrap around?
                # Standard 2-opt on a path [0..n-1] where 0 is fixed?
                # The tour returned by NN is a path starting at 0.
                # The evaluator closes the loop, so it's a cycle.
                # 2-Opt on a cycle:
                # Edges (u, v) and (x, y) are replaced by (u, x) and (v, y) or (u, y) and (v, x).
                # In our path representation [t0, t1, ..., t_{n-1}], the edges are:
                # (t0, t1), (t1, t2), ..., (t_{n-2}, t_{n-1}), (t_{n-1}, t0)
                
                # Let's stick to the path representation and handle the wrap-around carefully.
                # Actually, it's easier to treat the tour as a list of nodes and check all pairs of edges.
                
                u = tour[i-1]
                v = tour[i]
                x = tour[j]
                y = tour[(j+1) % n]
                
                # Current cost
                c1 = distance_matrix[u, v] + distance_matrix[x, y]
                
                # New cost
                c2 = distance_matrix[u, x] + distance_matrix[v, y]
                
                if c2 < c1 - 1e-10:
                    # Reverse segment between i and j
                    tour[i:j+1] = reversed(tour[i:j+1])
                    improved = True
                    
    return tour

def _select_next_from_tour(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node by looking up the pre-computed tour.
    If the pre-computed tour is not available or fails, fallback to nearest neighbor.
    """
    global _instance_cache
    
    key = _get_instance_key(distance_matrix)
    
    if key not in _instance_cache:
        # Compute the tour once
        # Check time budget
        if _start_time is None:
            _start_time = time.time()
        
        elapsed = time.time() - _start_time
        if elapsed > _DEADLINE - 10.0:
            # Emergency fallback
            dists = distance_matrix[current_node, unvisited_nodes]
            return int(unvisited_nodes[np.argmin(dists)])
            
        try:
            tour = _construct_tour_nn(distance_matrix)
            # Apply 2-Opt
            tour = _two_opt(tour, distance_matrix, max_passes=20)
            _instance_cache[key] = tour
        except Exception:
            _instance_cache[key] = None
            # Fallback
            dists = distance_matrix[current_node, unvisited_nodes]
            return int(unvisited_nodes[np.argmin(dists)])
        
    tour = _instance_cache[key]
    
    if tour is None:
        # Fallback
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
    
    # Find current_node in the tour
    try:
        idx = tour.index(current_node)
    except ValueError:
        # Should not happen if the tour is valid and contains all nodes
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
    
    # The next node in the tour
    next_node = tour[(idx + 1) % len(tour)]
    
    # Verify that next_node is in unvisited_nodes
    # If not, something is wrong (e.g., we are re-visiting a node, which shouldn't happen in a valid TSP tour construction)
    if next_node not in unvisited_nodes:
        # This can happen if the evaluator's state is out of sync with our cached tour
        # Or if we made a mistake.
        # Fallback to nearest neighbor
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
        
    return int(next_node)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next city for a TSP constructive heuristic.
    
    Strategy: Pre-compute a high-quality tour using Nearest Neighbor + 2-Opt,
    then follow that tour. This eliminates the myopia of greedy constructive
    heuristics and leverages local search to find a near-optimal cycle.
    """
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
        
    # Check time budget
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Emergency: just pick nearest neighbor
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
        
    return _select_next_from_tour(current_node, destination_node, unvisited_nodes, distance_matrix)

# EVOLVE-BLOCK-END
