import numpy as np
import time

def _two_opt(tour, distance_matrix, time_limit):
    """Improve a tour using 2-opt. Returns the improved tour."""
    n = len(tour)
    tour = list(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                n1, n2 = tour[i], tour[i + 1]
                n3, n4 = tour[j], tour[(j + 1) % n]
                
                d_old = distance_matrix[n1, n2] + distance_matrix[n3, n4]
                d_new = distance_matrix[n1, n3] + distance_matrix[n2, n4]
                
                if d_new < d_old - 1e-10:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
            if time.time() > time_limit:
                return tour
    return tour
    """Improve a tour using 2-opt. Returns the improved tour."""
    n = len(tour)
    tour = list(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                n1, n2 = tour[i], tour[i + 1]
                n3, n4 = tour[j], tour[(j + 1) % n]
                
                d_old = distance_matrix[n1, n2] + distance_matrix[n3, n4]
                d_new = distance_matrix[n1, n3] + distance_matrix[n2, n4]
                
                if d_new < d_old - 1e-10:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
            if time.time() > time_limit:
                return tour
    return tour

def _or_opt(tour, distance_matrix, time_limit):
    """Improve a tour using Or-Opt (moving segments of length 2 or 3)."""
    n = len(tour)
    tour = list(tour)
    improved = True
    while improved:
        improved = False
        for seg_len in [2, 3]:
            for i in range(n):
                # Segment to move: tour[i] ... tour[i+seg_len-1]
                # It is removed from position i and inserted at position j
                # Indices in the original tour before removal:
                # Segment: tour[i], ..., tour[i+seg_len-1]
                # Left neighbor: tour[(i-1) % n]
                # Right neighbor: tour[(i+seg_len) % n]
                
                # We try inserting this segment after every other node j
                # j should not be i, i-1, or i+seg_len (to avoid trivial moves or duplicates)
                # Also, we must be careful with wrap-around.
                
                start_idx = i
                end_idx = (i + seg_len) % n
                
                # If the segment wraps around, it's tricky. 
                # Standard Or-Opt usually linearizes or handles wrap carefully.
                # For simplicity and robustness, we skip segments that wrap around the end of the list
                # unless we rotate the tour first. 
                # Actually, rotating the tour so the segment doesn't wrap is a valid move.
                # But let's stick to simple non-wrapping segments for speed and simplicity in this context.
                # Since we can start the tour at any node, we can just rotate the tour so that i is not near the end.
                # However, modifying the tour structure in-place while iterating is dangerous.
                
                # Let's implement a safer version:
                # Try to move segment [i, i+seg_len) to position j.
                
                if i + seg_len > n:
                    continue # Skip wrapping segments for now, or rotate. 
                    # Rotating is better. Let's rotate the tour so that i is 0.
                    # But we are in a loop. Let's just skip for now, 2-opt is strong enough.
                    # Actually, for N=50, we can just rotate the list once before the loop?
                    # No, the tour changes.
                    # Let's just handle non-wrapping segments. This covers most moves.
                
                a = tour[(i - 1) % n]
                b = tour[i]
                c = tour[(i + seg_len) % n]
                
                # Cost of removing segment:
                # Old edges: (a, b) and (last_of_segment, c)
                # last_of_segment = tour[(i + seg_len - 1) % n]
                last_seg = tour[(i + seg_len - 1) % n]
                
                cost_remove = distance_matrix[a, b] + distance_matrix[last_seg, c]
                
                for j in range(n):
                    # j is the index of the node AFTER which we insert the segment.
                    # We don't want to insert it back where it was.
                    # The position "after j" corresponds to between j and (j+1)%n.
                    # If j == (i-1)%n, it's the original left neighbor.
                    # If j == (i+seg_len-1)%n, it's the original right neighbor (inserting at end of segment).
                    # We should skip these.
                    
                    if j == (i - 1) % n:
                        continue
                    if j == (i + seg_len - 1) % n:
                        continue
                    
                    # Also, if the insertion point is "inside" the segment being moved, skip.
                    # This happens if j is in [i, i+seg_len-1].
                    # Since we are iterating j from 0 to n-1, we need to check if j is in the segment range.
                    # This is tricky with wrap-around.
                    # Let's simplify: only try non-wrapping segments and non-wrapping insertions.
                    if i + seg_len > n:
                        continue
                    if j < i or j >= i + seg_len:
                        # j is outside the segment
                        pass
                    else:
                        continue
                    
                    d = tour[j]
                    e = tour[(j + 1) % n]
                    
                    # New edges: (a, e) is wrong. 
                    # We remove segment S. The gap is (a, c).
                    # We insert S between d and e.
                    # New edges: (a, c) [closing the gap], (d, b) [start of segment], (last_seg, e) [end of segment]
                    # Wait, the gap is closed by connecting a and c.
                    # The segment is inserted between d and e.
                    # So new edges are: (a, c), (d, b), (last_seg, e).
                    # Old edges were: (a, b), (last_seg, c), (d, e).
                    
                    cost_add = distance_matrix[a, c] + distance_matrix[d, b] + distance_matrix[last_seg, e]
                    cost_old = cost_remove + distance_matrix[d, e]
                    
                    if cost_add < cost_old - 1e-10:
                        # Perform the move
                        # 1. Extract segment
                        seg = tour[i : i + seg_len]
                        # 2. Remove from tour
                        del tour[i : i + seg_len]
                        # 3. Adjust j index if j > i
                        if j > i:
                            j -= seg_len
                        # 4. Insert after j
                        tour[j+1 : j+1] = seg
                        improved = True
                        break # Restart inner loop or just break and restart outer?
                        # Breaking inner loop is safer to restart j loop with new tour.
                if improved:
                    break
            if improved:
                break
            if time.time() > time_limit:
                break
    return tour

def _nearest_neighbor_tour(distance_matrix, start_node):
    """Generate a tour using nearest neighbor heuristic."""
    n = distance_matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start_node] = True
    tour = [start_node]
    current = start_node
    
    for _ in range(n - 1):
        dists = distance_matrix[current, :].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
        
    return tour

def heuristics(distance_matrix):
    start_time = time.time()
    n = distance_matrix.shape[0]
    
    # Time limit for the entire heuristic computation
    # Leave some margin before the 330s wall-clock limit
    # The heuristic is computed once per instance.
    # We have 3 instances. Total time for heuristics should be small.
    # Let's give 10 seconds per instance for local search.
    time_limit = start_time + 10.0
    
    # 1. Generate a high-quality reference tour using Greedy NN + 2-Opt + Or-Opt
    tour = _nearest_neighbor_tour(distance_matrix, 0)
    
    # 2-Opt improvement
    tour = _two_opt(tour, distance_matrix, time_limit)
    
    # Or-Opt improvement
    tour = _or_opt(tour, distance_matrix, time_limit)
    
    # 2. Build Heuristic Matrix
    # Base: 1 / d^2
    safe_dist = distance_matrix + 1e-9
    H = 1.0 / (safe_dist ** 2)
    
    # Boost edges in the reference tour
    # The tour is a cycle: tour[0]-tour[1]-...-tour[n-1]-tour[0]
    # We apply a stronger boost to guide the ants more effectively.
    # The ACO parameters (20 ants, 30 iters) are weak, so a strong heuristic is needed.
    boost_factor = 5.0
    
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        H[u, v] *= boost_factor
        H[v, u] *= boost_factor
        
    # Ensure positive
    H[H <= 0] = 1e-9
    
    return H
