import numpy as np
import time
import itertools

# Global cache for exact tail solutions? 
# The set of unvisited nodes changes every step, so caching is hard without keying on the frozenset.
# Given N=8 max for exact, we can just compute it.

_tour_start_time = None

def _solve_exact_tail(current_node, unvisited_nodes, destination_node, distance_matrix):
    """
    Solves the TSP path exactly for the remaining nodes.
    Returns the first node of the optimal path.
    Path: current -> p1 -> p2 -> ... -> pk -> destination
    """
    n = len(unvisited_nodes)
    if n == 0:
        return -1 # Should not happen
    if n == 1:
        return int(unvisited_nodes[0])
    
    # Generate permutations of unvisited_nodes
    # To speed up, we can use numpy for distance lookups inside the loop, 
    # but itertools.permutations is in Python.
    # For n=8, 40320 permutations.
    
    best_cost = float('inf')
    best_first = int(unvisited_nodes[0])
    
    # Optimization: If n is small, brute force.
    # We can prune if we want, but n<=8 is small enough.
    
    # Pre-fetch distances for these nodes to avoid repeated matrix indexing overhead?
    # dist_matrix is global in closure or passed.
    
    for perm in itertools.permutations(unvisited_nodes):
        cost = distance_matrix[current_node, perm[0]]
        for i in range(n - 1):
            cost += distance_matrix[perm[i], perm[i+1]]
        cost += distance_matrix[perm[-1], destination_node]
        
        if cost < best_cost:
            best_cost = cost
            best_first = int(perm[0])
            
    return best_first

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _tour_start_time
    
    # Deadline guard
    if _tour_start_time is None:
        _tour_start_time = time.time()
    
    elapsed = time.time() - _tour_start_time
    n_unvisited = len(unvisited_nodes)
    
    # If time is running out, fallback to simple NN
    if elapsed > 115:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    # Use exact solver for small tails
    # 8 nodes: 40320 perms. 
    # 9 nodes: 362880 perms. Might be slow in Python (0.1-0.5s).
    # Let's stick to 8 to be safe and fast.
    if n_unvisited <= 8:
        return _solve_exact_tail(current_node, unvisited_nodes, destination_node, distance_matrix)
    
    # Heuristic for larger sets: 2-step Lookahead
    # score(j) = d(current, j) + min(k in unvisited \ {j}) d(j, k)
    # This encourages picking nodes that are close to current AND have a close neighbor
    # in the remaining set, preventing "dead ends".
    
    best_score = float('inf')
    best_node = int(unvisited_nodes[0])
    
    # Vectorized calculation for the first part: d(current, unvisited)
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # For the min_next part, we need min distance from each j to other unvisited k.
    # We can compute the submatrix of distances between unvisited nodes.
    # sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    # Then for each row j, find min value excluding diagonal.
    
    # Since N is small (<=50), we can just loop or use numpy tricks.
    # Numpy trick:
    # Set diagonal to inf, then take min along axis 1.
    
    # Create mask or just index
    idx = np.arange(n_unvisited)
    
    # Get submatrix
    # This creates a copy, which is fine for N=50.
    sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Set diagonal to inf to ignore self-distance
    np.fill_diagonal(sub_dist, np.inf)
    
    # Min distance to any other node in unvisited set
    min_next_dist = np.min(sub_dist, axis=1)
    
    # Total score
    scores = dist_from_current + min_next_dist
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
