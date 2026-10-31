# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT_LIMIT = 140.0  # Leave 10s margin for 150s limit

def _is_timeout():
    return (time.time() - _START_TIME) > _TIMEOUT_LIMIT

def _tour_length(tour, distance_matrix):
    """Calculate the total length of a closed tour."""
    n = len(tour)
    if n == 0:
        return 0.0
    if n == 1:
        return 0.0
    # tour[0] -> tour[1] -> ... -> tour[n-1] -> tour[0]
    return np.sum(distance_matrix[tour[:-1], tour[1:]]) + distance_matrix[tour[-1], tour[0]]

def _nn_tour(nodes, distance_matrix):
    """Construct a tour using Nearest Neighbor heuristic."""
    n = len(nodes)
    if n == 0:
        return []
    if n == 1:
        return [nodes[0]]
    
    unvisited = set(nodes)
    current = nodes[0]
    tour = [current]
    unvisited.remove(current)
    
    while unvisited:
        # Find nearest neighbor among unvisited
        # Using numpy for speed
        candidates = list(unvisited)
        dists = distance_matrix[current, candidates]
        min_idx = np.argmin(dists)
        next_node = candidates[min_idx]
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node
        
    return tour

def _two_opt_cycle(tour, distance_matrix, deadline):
    """
    Apply 2-Opt improvement to a closed tour.
    Returns the improved tour.
    """
    n = len(tour)
    if n <= 3:
        return tour
    
    tour = tour.copy()
    
    # Pre-compute current cost for quick delta checks? 
    # Actually, calculating delta on the fly is O(1) per move.
    
    improved = True
    passes = 0
    max_passes = 100 # Safety limit
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        
        if time.time() > deadline - 0.5:
            break
            
        for i in range(n - 1):
            if time.time() > deadline - 0.5:
                break
                
            # j must be at least i+2 to avoid trivial moves
            # In a cycle, reversing a segment (i, j) removes edges (i, i+1) and (j, j+1 mod n)
            # and adds (i, j) and (i+1, j+1 mod n)
            for j in range(i + 2, n):
                # Avoid checking the reverse of the same move (i, j) and (j, i) isn't needed 
                # if we just break on improvement, but standard 2-opt usually checks all pairs.
                # To be efficient, we can just check all j > i+1.
                
                u = tour[i]
                v = tour[(i + 1) % n]
                x = tour[j]
                y = tour[(j + 1) % n]
                
                d_uv = distance_matrix[u, v]
                d_xy = distance_matrix[x, y]
                d_ux = distance_matrix[u, x]
                d_vy = distance_matrix[v, y]
                
                delta = d_ux + d_vy - d_uv - d_xy
                
                if delta < -1e-10:
                    # Reverse segment from i+1 to j
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
                    # Break inner loop to restart with new structure? 
                    # Or continue? Breaking inner is standard for simple implementation.
                    # Restarting outer is safer but slower. Let's break inner and continue outer 
                    # but we might miss improvements. 
                    # Standard "first improvement" strategy: break inner, continue outer.
                    break
                    
            if improved:
                # If we made a change, restarting the outer loop is often better to ensure 
                # we don't miss cascading improvements, but it's slower.
                # Let's stick to breaking inner and continuing outer for speed, 
                # but this might get stuck in local minima faster.
                # Actually, breaking outer and restarting is more robust for quality.
                # Given N=50, restarting is cheap.
                break
                
    return tour

def _or_opt_cycle(tour, distance_matrix, deadline):
    """
    Apply Or-Opt moves (move 1 or 2 cities) to a closed tour.
    """
    n = len(tour)
    if n <= 4:
        return tour
    
    tour = tour.copy()
    improved = True
    passes = 0
    max_passes = 50
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        
        if time.time() > deadline - 0.5:
            break
            
        # Try moving 1 city
        for i in range(n):
            if time.time() > deadline - 0.5:
                break
                
            city = tour[i]
            
            # Try inserting it at position j (j != i, j != i-1, j != i+1)
            # We remove city from i, then insert at j.
            # If we remove from i, indices shift. 
            # Easier: define move as removing segment [i:i+1] and inserting at [j:j]
            
            for j in range(n):
                if j == i or j == (i + 1) % n or j == (i - 1) % n:
                    continue
                    
                # Calculate delta
                # Remove city from i:
                # Edges removed: (i-1, i) and (i, i+1)
                # Edges added: (i-1, i+1)
                
                # Insert city at j (in the modified list? No, let's think in terms of original indices)
                # This is tricky with in-place modification. 
                # Let's use a helper to calculate cost of a specific move.
                
                # Actually, Or-Opt is complex to implement efficiently with in-place reversal logic 
                # without careful index management. 
                # Given N=50, we can just reconstruct the tour for each candidate move? No, too slow.
                # Let's stick to 2-Opt which is robust and fast. Or-Opt often provides marginal gains 
                # for N=50 if 2-Opt is well-implemented. 
                # However, the inspirations mention Or-Opt. Let's implement a simple version.
                
                # For simplicity and robustness within the time limit, I will skip complex Or-Opt 
                # and rely on multiple 2-Opt passes with random restarts or just very thorough 2-Opt.
                # Actually, let's implement a simple "insertion" based Or-Opt:
                # Remove node i. Find best place to re-insert it.
                
                # This requires O(N^2) checks per node, so O(N^3) total. 50^3 = 125,000 ops. Fast enough.
                
                # Let's implement a simpler "1-move" Or-Opt:
                # Remove node at i. 
                # The tour is now n-1 nodes.
                # Find best insertion point for node i.
                
                # This is effectively "Reinsertion".
                # Cost of removing i:
                # d(i-1, i) + d(i, i+1) - d(i-1, i+1)
                
                # Cost of inserting i at j (between j-1 and j in the reduced tour):
                # We need to map indices carefully.
                
                # Given the complexity and risk of bugs, and that 2-Opt is very effective, 
                # I will focus on a robust 2-Opt with a "first improvement" vs "best improvement" 
                # strategy.
                pass
                
        # If we don't implement Or-Opt properly, we might miss gains. 
        # Let's try a different approach: Multiple 2-Opt passes.
        # The current _two_opt_cycle breaks on first improvement. 
        # Let's ensure we run multiple passes.
        
    return tour

def _compute_best_tour(nodes, distance_matrix, deadline):
    """
    Compute a high-quality tour using NN + 2-Opt.
    """
    nodes = list(nodes)
    
    # Generate initial tour
    tour = _nn_tour(nodes, distance_matrix)
    
    # Improve with 2-Opt
    # We can run 2-Opt multiple times.
    # Since 2-Opt converges, running it once to local optimum is good.
    # To escape local minima, we could do random restarts, but NN is usually decent.
    # Let's do a few passes of 2-Opt.
    
    for _ in range(3):
        if time.time() > deadline - 1.0:
            break
        new_tour = _two_opt_cycle(tour, distance_matrix, deadline)
        if _tour_length(new_tour, distance_matrix) < _tour_length(tour, distance_matrix):
            tour = new_tour
        else:
            break
            
    return tour

# Cache for pre-computed tours
_tour_cache = {}
_tour_cache_lock = None # No threading, so no lock needed

def _get_instance_id(distance_matrix):
    """
    Create a simple hash of the distance matrix to identify the instance.
    """
    # Using the first row and last row and a few random elements as a fingerprint
    # Since it's a deterministic instance, we can just hash the whole thing if small, 
    # but 50x50 is 2500 floats. Hashing the array is fast enough.
    return hash(distance_matrix.tobytes())

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in a TSP tour.
    Strategy: Pre-compute the optimal (or near-optimal) tour on the first call.
    Subsequent calls just return the next node in that tour.
    """
    
    if _is_timeout():
        # Fallback to nearest neighbor if we are running out of time
        if len(unvisited_nodes) == 0:
            return int(current_node)
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])

    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        n1, n2 = unvisited_nodes
        cost1 = distance_matrix[current_node, n1] + distance_matrix[n1, n2] + distance_matrix[n2, destination_node]
        cost2 = distance_matrix[current_node, n2] + distance_matrix[n2, n1] + distance_matrix[n1, destination_node]
        return int(n1 if cost1 <= cost2 else n2)

    # Identify the instance
    # We need to know the full set of nodes to compute the tour.
    # The full set of nodes is usually 0..N-1. 
    # We can infer N from the size of distance_matrix.
    N = distance_matrix.shape[0]
    all_nodes = np.arange(N)
    
    # Check if we have already computed the tour for this instance
    inst_id = _get_instance_id(distance_matrix)
    
    if inst_id not in _tour_cache:
        # Compute the tour
        # Deadline: leave enough time for the rest of the calls.
        # The calls are fast, so we can use most of the time for computation.
        # However, we must not exceed the global timeout.
        # Let's set a deadline for this computation.
        comp_deadline = _START_TIME + _TIMEOUT_LIMIT - 5.0 # Leave 5s for safety
        
        if time.time() > comp_deadline:
            # Not enough time to compute a good tour, fallback to NN
            tour = _nn_tour(all_nodes, distance_matrix)
        else:
            tour = _compute_best_tour(all_nodes, distance_matrix, comp_deadline)
            
        _tour_cache[inst_id] = tour
        
    else:
        tour = _tour_cache[inst_id]
        
    # Now, find the next node in the tour after current_node
    # The tour is a list of node indices.
    # We need to find the index of current_node in the tour, and return the next one.
    # Note: The tour is a cycle.
    
    # Optimization: Pre-compute a mapping from node to its successor in the tour
    # We can store the successor map in the cache too, or just compute it.
    # Computing it on the fly is O(N). N=50, so it's fast.
    
    # Find current_node in tour
    try:
        idx = tour.index(current_node)
    except ValueError:
        # Should not happen if current_node is in the instance
        # Fallback to NN
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
        
    next_idx = (idx + 1) % N
    next_node = tour[next_idx]
    
    # The next node must be in unvisited_nodes. 
    # If it's not, something is wrong (e.g., we visited it already, which shouldn't happen 
    # in a proper TSP construction where we remove visited nodes).
    # However, the interface implies we are building a tour step-by-step.
    # If the pre-computed tour suggests a node that is NOT in unvisited_nodes, 
    # it means we have already visited it, which is a contradiction if we follow the tour strictly.
    # Wait, the constructive heuristic visits nodes one by one. 
    # If we follow the pre-computed tour, we will visit each node exactly once.
    # So next_node should always be in unvisited_nodes, unless we made a mistake 
    # or the "current_node" is not part of the tour we computed (e.g., if the instance changed).
    
    if next_node not in unvisited_nodes:
        # This can happen if the tour computation was for a different set of nodes, 
        # or if we are re-entering a state. 
        # Fallback to nearest neighbor among unvisited.
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
        
    return int(next_node)

# EVOLVE-BLOCK-END
