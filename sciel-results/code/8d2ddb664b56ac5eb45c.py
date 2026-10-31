"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level cache for reference tours and deadlines
# Key: (id of distance matrix, shape, first element hash) -> dict with 'tour', 'successor', 'deadline'
_cache = {}
_DEADLINE_MARGIN = 15  # seconds before the 150s kill

def _get_reference_tour(distance_matrix):
    """
    Compute a high-quality reference tour using multi-start NN + 2-opt.
    Caches the result based on the distance matrix.
    """
    # Create a simple hash key for the matrix
    # We can't hash the whole matrix easily if it's large, but 50x50 is small.
    # Let's use a tuple of the first row and shape as a proxy, or just id if we assume same object.
    # But the evaluator might pass different arrays with same content.
    # Let's use a quick checksum.
    key = (distance_matrix.shape, distance_matrix[0,0], distance_matrix[0,1], distance_matrix[1,0], distance_matrix[1,1], distance_matrix[49,49] if distance_matrix.shape[0]>=50 else distance_matrix[-1,-1])
    
    if key in _cache:
        return _cache[key]
    
    n = distance_matrix.shape[0]
    best_tour = None
    best_length = float('inf')
    
    # Multi-start NN
    # Try starting from a few different nodes
    start_nodes = [0, n//2, n-1]
    # Maybe add a few random ones if time permits, but let's keep it deterministic and fast
    # Actually, for 50 nodes, even 5-10 starts are fast.
    
    for start in start_nodes:
        tour = _nn_tour(start, n, distance_matrix)
        tour, length = _two_opt(tour, distance_matrix)
        if length < best_length:
            best_length = length
            best_tour = tour
    
    # Store successor mapping for fast lookup
    successor = {}
    for i in range(len(best_tour)):
        curr = best_tour[i]
        nxt = best_tour[(i+1) % len(best_tour)]
        successor[curr] = nxt
    
    # Also store the tour itself for potential fallback
    _cache[key] = {'tour': best_tour, 'successor': successor, 'length': best_length}
    return _cache[key]

def _nn_tour(start, n, distance_matrix):
    """Construct a tour using the Nearest Neighbor heuristic starting from 'start'."""
    visited = np.zeros(n, dtype=bool)
    tour = [start]
    visited[start] = True
    current = start
    
    for _ in range(n - 1):
        # Find nearest unvisited node
        unvisited = np.where(~visited)[0]
        dists = distance_matrix[current, unvisited]
        nearest_idx = np.argmin(dists)
        next_node = unvisited[nearest_idx]
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    
    return tour

def _two_opt(tour, distance_matrix):
    """Improve a tour using the 2-opt heuristic."""
    n = len(tour)
    if n < 4:
        return tour, _tour_length(tour, distance_matrix)
    
    improved = True
    # Limit iterations to ensure speed
    max_iters = 100
    
    while improved and max_iters > 0:
        improved = False
        max_iters -= 1
        
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Skip if edges are adjacent in the tour
                if j == i + 1:
                    continue
                # If i=0 and j=n-1, they are also adjacent (closing the loop)
                if i == 0 and j == n - 1:
                    continue
                
                # Calculate delta
                # Edge (i, i+1) and (j, j+1) are replaced by (i, j) and (i+1, j+1)
                # Note: indices in tour, not node IDs
                u = tour[i]
                v = tour[i + 1]
                w = tour[j]
                x = tour[(j + 1) % n]
                
                # Current length of these two edges
                current_len = distance_matrix[u, v] + distance_matrix[w, x]
                # New length
                new_len = distance_matrix[u, w] + distance_matrix[v, x]
                
                if new_len < current_len - 1e-10:
                    # Reverse the segment from i+1 to j
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    
    return tour, _tour_length(tour, distance_matrix)

def _tour_length(tour, distance_matrix):
    """Calculate the total length of a tour."""
    n = len(tour)
    length = 0.0
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        length += distance_matrix[u, v]
    return length

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node for the TSP tour.
    Uses a reference tour (NN + 2-opt) to guide the choice.
    If the reference successor is in unvisited_nodes, pick it.
    Otherwise, pick the unvisited node with the lowest combined cost:
    dist(current, candidate) + dist(candidate, destination)
    This is a "look-ahead" or "insertion-like" heuristic that avoids greedy myopia.
    """
    # Deadline guard
    # We don't have a global start time, so we can't easily enforce a strict per-call deadline
    # unless we track it. However, the computation here is very fast (O(N) lookup or O(N) scan).
    # The expensive part is _get_reference_tour, which is cached.
    # If the cache miss happens, it takes ~0.1-0.5s for 50 nodes.
    # This is well within the 150s limit for 8 instances.
    
    ref_data = _get_reference_tour(distance_matrix)
    successor = ref_data['successor']
    
    # Check if the reference successor is available
    ref_next = successor.get(current_node, None)
    if ref_next is not None and ref_next in unvisited_nodes:
        return int(ref_next)
    
    # Fallback: Choose node that minimizes dist(current, node) + dist(node, destination)
    # This approximates the cost of inserting the node between current and destination.
    if len(unvisited_nodes) == 0:
        return int(current_node) # Should not happen per contract
        
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
        
    # Vectorized cost calculation
    # dist_current = distance_matrix[current_node, unvisited_nodes]
    # dist_dest = distance_matrix[destination_node, unvisited_nodes]
    # costs = dist_current + dist_dest
    
    # Note: distance_matrix is symmetric, so dist(node, dest) == dist(dest, node)
    # We can just use the row for destination_node.
    
    costs = distance_matrix[current_node, unvisited_nodes] + distance_matrix[destination_node, unvisited_nodes]
    best_idx = np.argmin(costs)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
