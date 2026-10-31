"""
Strategy: Global Pre-computation (Lookahead).
Instead of making greedy local choices, we compute a near-optimal closed tour 
for the entire set of cities on the first call and then simply traverse it.
This avoids the "greedy trap" where early local choices force expensive 
connections later.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

_START_TIME = time.time()
TIME_LIMIT = 150.0
# We need to be safe. The evaluator kills at 150s. 
# We should aim to finish all work well before that.
# Since we do heavy lifting in the first call, we must be very fast there.
# Subsequent calls are O(1).

# Cache for pre-computed tours
# Key: fingerprint of the distance matrix (e.g., hash of flattened array)
# Value: list of nodes in the optimal tour order
_TOUR_CACHE = {}

def _get_instance_id(distance_matrix):
    """
    Create a unique identifier for the distance matrix to cache the tour.
    Using a hash of the flattened array is robust.
    """
    try:
        # Flatten and take a hash. For 50x50, this is small enough.
        flat = distance_matrix.flatten()
        # Use a simple hash to avoid storing the whole array as key
        return hash(flat.tobytes())
    except Exception:
        return "default"

def _tour_length(tour, distance_matrix):
    """Calculate the total length of a closed tour."""
    n = len(tour)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        total += distance_matrix[tour[i], tour[(i + 1) % n]]
    return total

def _nn_tour(nodes, distance_matrix):
    """
    Construct a tour using Nearest Neighbor heuristic.
    Starts at node 0.
    """
    n = len(nodes)
    if n == 0:
        return []
    if n == 1:
        return [nodes[0]]
    
    # Work with indices relative to the 'nodes' list or absolute?
    # Let's work with absolute node IDs from the input list.
    # Input 'nodes' is a list of city IDs.
    visited = [False] * len(nodes)
    # Map city ID to index in 'nodes' for quick lookup? 
    # Or just use the distance_matrix directly with city IDs.
    # distance_matrix is indexed by city ID (0..N-1).
    
    current = nodes[0]
    tour = [current]
    visited_idx = set([current])
    
    remaining = [x for x in nodes if x != current]
    
    while remaining:
        # Find nearest unvisited
        best_node = -1
        best_dist = float('inf')
        # Vectorized approach for speed
        # Get distances from current to all remaining
        dists = distance_matrix[current, remaining]
        min_idx = np.argmin(dists)
        best_node = remaining[min_idx]
        
        tour.append(best_node)
        visited_idx.add(best_node)
        remaining.remove(best_node)
        current = best_node
        
    return tour

def _two_opt(tour, distance_matrix, max_iters=100):
    """
    Improve a closed tour using 2-Opt.
    """
    n = len(tour)
    if n < 4:
        return tour
    
    tour = list(tour)
    improved = True
    iter_count = 0
    
    # Precompute distances? No, just use matrix.
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        for i in range(n - 1):
            if i + 1 >= n: break
            a = tour[i]
            b = tour[i + 1]
            for j in range(i + 2, n):
                if j == 0: continue # Closed tour, avoid crossing start/end in a way that breaks logic easily
                # Actually, standard 2-opt on a cycle:
                # Reverse segment i+1 to j
                c = tour[j]
                d = tour[(j + 1) % n]
                
                # Current edges: (a,b) and (c,d)
                # New edges: (a,c) and (b,d)
                old_cost = distance_matrix[a, b] + distance_matrix[c, d]
                new_cost = distance_matrix[a, c] + distance_matrix[b, d]
                
                if new_cost < old_cost - 1e-10:
                    # Reverse segment i+1 to j
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    improved = True
                    # Break inner loop to restart? Or continue?
                    # Usually restart is safer for correctness, but slower.
                    # For N=50, it's fast.
                    break
            if improved:
                break
    return tour

def _or_opt(tour, distance_matrix, max_iters=50):
    """
    Improve a closed tour using Or-Opt (moves a segment of 1, 2, or 3 nodes).
    """
    n = len(tour)
    if n < 5:
        return tour
    
    tour = list(tour)
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        
        # Try moving segments of size 1, 2, 3
        for size in [1, 2, 3]:
            if n <= size + 1:
                continue
            for i in range(n):
                # Segment to move: tour[i : i+size]
                # It is between tour[i-1] and tour[i+size]
                prev_node = tour[(i - 1) % n]
                next_node = tour[(i + size) % n]
                
                segment = tour[i : i+size]
                # The segment is removed. The new connection is prev_node -> next_node.
                
                # Try inserting the segment at all other positions
                for j in range(n):
                    # Skip if insertion is at the same place or adjacent (no change)
                    if j == i or j == (i - 1) % n or j == (i + size) % n or j == (i + size - 1) % n:
                        continue
                    # Also avoid inserting into the segment itself
                    if j >= i and j < i + size:
                        continue
                    if j == (i + size) % n:
                        continue
                        
                    # Insertion position: between tour[j-1] and tour[j]
                    # Note: j is the index where the segment will end up before.
                    # So it goes between (j-1) and j.
                    # Wait, if we insert before index j, it goes between j-1 and j.
                    
                    # Let's define insertion more clearly.
                    # We remove segment S from position i.
                    # We insert S before index k in the *remaining* list?
                    # It's easier to work on the full list.
                    
                    # Let's stick to a simpler Or-Opt implementation:
                    # Move 1 node.
                    if size == 1:
                        node = tour[i]
                        # Remove node
                        tour.pop(i)
                        # Now n is n-1
                        n_temp = len(tour)
                        
                        best_pos = -1
                        best_cost = float('inf')
                        
                        # Try all positions
                        for k in range(n_temp):
                            # Insert before k
                            # Neighbors in new tour: prev = (k-1)%n_temp, next = k
                            prev_k = tour[(k - 1) % n_temp]
                            next_k = tour[k]
                            
                            # Old edges removed: (prev_node, next_node) -- wait, prev_node and next_node are from the ORIGINAL tour before removal.
                            # Let's re-calculate based on current state.
                            # This is getting complex. Let's use a standard robust Or-Opt.
                            
                        # Revert
                        tour.insert(i, node)
                        n = len(tour)
                        
                        # Let's implement a simpler "Move 1 node" specifically.
                        # Cost of removing node X between A and B: dist(A,X) + dist(X,B) - dist(A,B)
                        # Cost of inserting X between C and D: dist(C,X) + dist(X,D) - dist(C,D)
                        # Gain = (dist(A,X) + dist(X,B) - dist(A,B)) - (dist(C,X) + dist(X,D) - dist(C,D))
                        # We want Gain > 0.
                        
                        A = tour[(i - 1) % n]
                        B = tour[(i + 1) % n]
                        X = tour[i]
                        
                        remove_cost = distance_matrix[A, X] + distance_matrix[X, B] - distance_matrix[A, B]
                        
                        for j in range(n):
                            if j == i or j == (i - 1) % n or j == (i + 1) % n:
                                continue
                            C = tour[(j - 1) % n]
                            D = tour[j]
                            # Insert X between C and D
                            insert_cost = distance_matrix[C, X] + distance_matrix[X, D] - distance_matrix[C, D]
                            
                            gain = remove_cost - insert_cost
                            if gain > 1e-10:
                                # Perform move
                                tour.pop(i)
                                # Now insert at position j. 
                                # Note: if j > i, the index j shifts by -1 after pop.
                                # If j < i, index j is unchanged.
                                # We want to insert BEFORE the node currently at index j in the *original* list?
                                # No, we want to insert between C and D.
                                # C is at (j-1) % n in original.
                                # D is at j in original.
                                # After popping i:
                                # If j > i: C is at j-2, D is at j-1. Insert at j-1.
                                # If j < i: C is at j-1, D is at j. Insert at j.
                                
                                if j > i:
                                    insert_idx = j - 1
                                else:
                                    insert_idx = j
                                    
                                tour.insert(insert_idx, X)
                                improved = True
                                # Break out of loops to restart
                                break
                        if improved:
                            break
                    # Size 2 and 3 are similar but more complex. Given N=50, 2-Opt is very powerful.
                    # Or-Opt adds marginal gain but complexity. Let's stick to 2-Opt + NN for reliability and speed.
                    # If 2-Opt is fast enough, we can run multiple passes or combine with Or-Opt later.
                    # For now, let's add a simple Or-Opt for size 1 only, as it's the most common improvement.
                    
            if improved:
                break
                
    return tour

def _compute_best_tour(nodes, distance_matrix):
    """
    Compute a high-quality tour.
    1. NN construction
    2. 2-Opt improvement
    3. Or-Opt (size 1) improvement
    """
    # Ensure nodes are a list of ints
    nodes = list(nodes)
    
    # 1. NN
    tour = _nn_tour(nodes, distance_matrix)
    
    # 2. 2-Opt
    # Use a time check? The first call has the most time.
    # Let's run a fixed number of 2-Opt passes.
    for _ in range(5):
        tour = _two_opt(tour, distance_matrix, max_iters=50)
        
    # 3. Or-Opt
    for _ in range(5):
        tour = _or_opt(tour, distance_matrix, max_iters=20)
        
    return tour

def _get_tour(nodes, distance_matrix):
    """
    Retrieve or compute the best tour for the given nodes.
    """
    key = _get_instance_id(distance_matrix)
    # Also include the set of nodes in case the matrix is the same but subset differs?
    # The task says "eight seeded uniform 50-city instances".
    # So the matrix defines the instance. The nodes are likely always all 50.
    # But to be safe, let's include a hash of the sorted nodes.
    node_key = hash(tuple(sorted(nodes)))
    full_key = (key, node_key)
    
    if full_key in _TOUR_CACHE:
        return _TOUR_CACHE[full_key]
    
    # Compute
    tour = _compute_best_tour(nodes, distance_matrix)
    _TOUR_CACHE[full_key] = tour
    
    return tour

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node by following a pre-computed near-optimal tour.
    """
    U = np.asarray(unvisited_nodes, dtype=int)
    if len(U) == 0:
        return -1
    if len(U) == 1:
        return int(U[0])

    # Time check: if we are about to timeout, fall back to fast nearest neighbor
    if time.time() - _START_TIME > TIME_LIMIT - 5.0:
        # Emergency fallback
        d = distance_matrix[current_node, U]
        return int(U[np.argmin(d)])

    try:
        # Get the optimal tour for all unvisited nodes + current node?
        # The tour should cover all nodes that will be visited.
        # The 'unvisited_nodes' are the ones remaining.
        # We have already visited some nodes.
        # The pre-computed tour is for the FULL set of 50 nodes.
        # We need to find where 'current_node' is in the tour, and then follow the tour
        # skipping any nodes that are NOT in unvisited_nodes (i.e., already visited).
        
        # Wait, the 'unvisited_nodes' passed in changes as the tour progresses.
        # On the first call, unvisited_nodes is all nodes except the start.
        # On subsequent calls, it shrinks.
        # The distance_matrix is constant for the instance.
        
        # So we should compute the tour for the FULL set of nodes in the first call.
        # How do we know the full set? 
        # We can infer it from the first call's unvisited_nodes + current_node.
        # Let's store the full node set.
        
        if not hasattr(select_next_node, '_full_nodes'):
            # First call
            full_nodes = list(U) + [int(current_node)]
            # Ensure unique
            full_nodes = list(set(full_nodes))
            select_next_node._full_nodes = full_nodes
            
            # Compute tour for full nodes
            tour = _compute_best_tour(full_nodes, distance_matrix)
            select_next_node._tour = tour
            select_next_node._tour_set = set(tour)
            
        else:
            tour = select_next_node._tour
            full_nodes = select_next_node._full_nodes

        # Find current_node in the tour
        try:
            idx = tour.index(int(current_node))
        except ValueError:
            # If current_node is not in the tour (shouldn't happen), fallback
            d = distance_matrix[current_node, U]
            return int(U[np.argmin(d)])

        # Traverse the tour from idx+1 onwards
        n = len(tour)
        for k in range(1, n + 1):
            candidate = tour[(idx + k) % n]
            if candidate in U:
                return int(candidate)
        
        # Fallback if something went wrong
        d = distance_matrix[current_node, U]
        return int(U[np.argmin(d)])

    except Exception:
        # Fallback to nearest neighbor
        d = distance_matrix[current_node, U]
        return int(U[np.argmin(d)])

# EVOLVE-BLOCK-END
