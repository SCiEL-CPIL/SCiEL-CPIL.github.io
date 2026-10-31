"""Combined strategy: Pre-compute high-quality tour using NN + Greedy Insertion + 2-Opt + Or-Opt."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state
_DEADLINE = None
_CACHE = {}
_START_TIME = None

def _get_instance_id(distance_matrix):
    """Create a lightweight fingerprint for the distance matrix."""
    n = distance_matrix.shape[0]
    if n > 10:
        sample = distance_matrix[::max(1, n//10), :].tobytes()
    else:
        sample = distance_matrix.tobytes()
    return hash(sample)

def _tour_length(tour, distance_matrix):
    """Calculate the total length of the tour."""
    n = len(tour)
    if n == 0:
        return 0.0
    return np.sum(distance_matrix[tour[:-1], tour[1:]]) + distance_matrix[tour[-1], tour[0]]

def _two_opt(tour, distance_matrix, deadline):
    """
    Apply 2-Opt improvement to a tour.
    tour: numpy array of node indices
    """
    n = len(tour)
    if n <= 3:
        return tour
    
    tour = tour.copy()
    current_cost = _tour_length(tour, distance_matrix)
    
    improved = True
    passes = 0
    max_passes = 1000
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        
        if time.time() > deadline - 2.0:
            break
            
        for i in range(n - 1):
            if time.time() > deadline - 2.0:
                break
                
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                    
                u = tour[i]
                v = tour[i+1]
                x = tour[j]
                y = tour[(j+1) % n]
                
                d_uv = distance_matrix[u, v]
                d_xy = distance_matrix[x, y]
                d_ux = distance_matrix[u, x]
                d_vy = distance_matrix[v, y]
                
                delta = d_ux + d_vy - d_uv - d_xy
                
                if delta < -1e-10:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    current_cost += delta
                    improved = True
                    # Restart inner loop? No, just continue. 
                    # To be efficient, we could break and restart outer, but simple continue is okay for N=50.
                    # Actually, breaking inner and restarting outer is safer for convergence speed in some cases.
                    break
            
            if improved:
                # If we found an improvement, we break the inner loop.
                # We do NOT break the outer loop immediately unless we want to be very strict.
                # Standard 2-opt usually restarts from i=0 after a change to ensure no crossing edges are missed.
                # However, for speed, we can just continue. Let's break outer to restart from 0 for better quality.
                break
                
    return tour

def _or_opt(tour, distance_matrix, deadline):
    """
    Apply Or-Opt moves (segments of length 1, 2, 3).
    """
    n = len(tour)
    if n <= 4:
        return tour
        
    tour = tour.copy()
    current_cost = _tour_length(tour, distance_matrix)
    
    improved = True
    passes = 0
    max_passes = 100
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        
        if time.time() > deadline - 2.0:
            break
            
        for seg_len in [1, 2, 3]:
            if time.time() > deadline - 2.0:
                break
                
            for i in range(n):
                if time.time() > deadline - 2.0:
                    break
                    
                # Segment to move: tour[i : i+seg_len] (circular)
                # To handle circularity simply, we duplicate tour or use modulo.
                # Let's extract the segment and its neighbors.
                
                # Get segment indices
                seg_indices = [(i + k) % n for k in range(seg_len)]
                seg = tour[seg_indices]
                
                # Neighbors
                prev_idx = (i - 1) % n
                next_idx = (i + seg_len) % n
                
                # If the segment wraps around 0, the logic is slightly different for removal.
                # To simplify, let's only consider non-wrapping segments in the main loop 
                # and handle the wrap-around case separately or just rely on 2-opt for those?
                # Actually, Or-Opt is powerful. Let's implement it carefully.
                
                # For simplicity and robustness in this time-constrained script, 
                # we will only move segments that do NOT wrap around the end of the array.
                # This covers most cases effectively.
                if i + seg_len > n:
                    continue

                prev_node = tour[prev_idx]
                next_node = tour[next_idx]
                
                # Cost of edges removed: (prev, seg[0]) and (seg[-1], next)
                # Internal edges of seg remain, so we don't need to calculate them for delta.
                d_prev_seg0 = distance_matrix[prev_node, seg[0]]
                d_seg_last_next = distance_matrix[seg[-1], next_node]
                
                # Try inserting this segment at position j
                # We iterate j from 0 to n - seg_len
                best_delta = 0
                best_j = -1
                
                for j in range(n - seg_len + 1):
                    # Skip positions that overlap with the original segment
                    if j <= i < j + seg_len or i <= j < i + seg_len:
                        # More precise overlap check:
                        # Original range: [i, i+seg_len-1]
                        # New range: [j, j+seg_len-1]
                        if max(i, j) < min(i + seg_len, j + seg_len):
                            continue
                            
                    # Insertion position: between tour[j-1] and tour[j]
                    # If j == 0, insert at beginning (between end and start)
                    # If j > 0, insert between j-1 and j
                    
                    if j == 0:
                        new_prev = tour[n-1]
                        new_next = tour[0]
                    else:
                        new_prev = tour[j-1]
                        new_next = tour[j]
                        
                    # Cost of edges added: (new_prev, seg[0]) and (seg[-1], new_next)
                    d_new_prev_seg0 = distance_matrix[new_prev, seg[0]]
                    d_new_seg_last_next = distance_matrix[seg[-1], new_next]
                    
                    # The edge (new_prev, new_next) is replaced by the segment.
                    # So we remove (new_prev, new_next) and add the two new edges.
                    # Wait, the standard Or-Opt delta calculation:
                    # Old: (prev, seg0) + (segLast, next) + (new_prev, new_next)
                    # New: (new_prev, seg0) + (segLast, new_next) + (prev, next)
                    # Delta = New - Old
                    
                    d_new_prev_next = distance_matrix[new_prev, new_next]
                    d_prev_next = distance_matrix[prev_node, next_node]
                    
                    delta = (d_new_prev_seg0 + d_new_seg_last_next + d_prev_next) - \
                            (d_prev_seg0 + d_seg_last_next + d_new_prev_next)
                    
                    if delta < best_delta - 1e-10:
                        best_delta = delta
                        best_j = j
                        
                if best_j != -1:
                    # Perform the move
                    # 1. Remove segment from [i, i+seg_len)
                    # Since we ensured no wrap, we can do this with array slicing
                    tour = np.concatenate([tour[:i], tour[i+seg_len:]])
                    n_new = len(tour)
                    
                    # 2. Adjust insertion index if it was after the removed segment
                    if best_j > i:
                        best_j -= seg_len
                    # If best_j was 0 and we removed from 0, it stays 0.
                    # If best_j was after i, it shifts left.
                    # If best_j was before i, it stays same.
                    # Note: best_j in the loop was an index in the ORIGINAL tour.
                    # If best_j > i, the segment [0..i-1] remains, then [i+seg_len..n-1] shifts to [i..n-seg_len-1].
                    # So indices > i decrease by seg_len.
                    
                    # 3. Insert segment at best_j
                    if best_j == 0:
                        tour = np.concatenate([seg, tour])
                    else:
                        tour = np.concatenate([tour[:best_j], seg, tour[best_j:]])
                    
                    current_cost += best_delta
                    improved = True
                    break # Restart the i loop with the new tour
                    
            if improved:
                break # Restart the seg_len loop with the new tour
                
    return tour

def _construct_nn_tour(distance_matrix, start):
    """Construct a tour using Nearest Neighbor heuristic."""
    n = distance_matrix.shape[0]
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    curr = start
    
    while unvisited:
        # Find nearest unvisited node
        dists = distance_matrix[curr, list(unvisited)]
        min_idx = np.argmin(dists)
        next_node = list(unvisited)[min_idx]
        tour.append(next_node)
        unvisited.remove(next_node)
        curr = next_node
        
    return np.array(tour)

def _construct_greedy_insertion_tour(distance_matrix, start):
    """Construct a tour using Greedy Insertion heuristic."""
    n = distance_matrix.shape[0]
    # Start with a single node
    tour = [start]
    unvisited = set(range(n))
    unvisited.remove(start)
    
    while unvisited:
        best_node = None
        best_pos = None
        best_delta = np.inf
        
        # Find the node that can be inserted with minimum cost increase
        for node in unvisited:
            # Find best position to insert 'node' into 'tour'
            min_delta = np.inf
            min_pos = 0
            
            for i in range(len(tour)):
                prev_node = tour[i]
                next_node = tour[(i + 1) % len(tour)]
                
                # Cost of inserting node between prev_node and next_node
                delta = distance_matrix[prev_node, node] + distance_matrix[node, next_node] - distance_matrix[prev_node, next_node]
                
                if delta < min_delta:
                    min_delta = delta
                    min_pos = i + 1 # Insert after i
            
            if min_delta < best_delta:
                best_delta = min_delta
                best_node = node
                best_pos = min_pos
                
        # Insert best_node at best_pos
        tour.insert(best_pos, best_node)
        unvisited.remove(best_node)
        
    return np.array(tour)

def _refine_tour(tour, distance_matrix, deadline):
    """Apply 2-Opt and Or-Opt to refine a tour."""
    # Sequence: 2-Opt -> Or-Opt -> 2-Opt
    # This helps escape local minima
    tour = _two_opt(tour, distance_matrix, deadline)
    tour = _or_opt(tour, distance_matrix, deadline)
    tour = _two_opt(tour, distance_matrix, deadline)
    return tour

def _construct_best_tour(distance_matrix, deadline):
    """
    Generate multiple candidate tours using different heuristics and starting points,
    refine them, and return the best one.
    """
    n = distance_matrix.shape[0]
    best_tour = None
    best_len = np.inf
    
    # Define starting points. For N=50, we can try all or a subset.
    # Let's try all starting points for NN and a subset for Greedy Insertion (which is slower).
    
    # 1. Nearest Neighbor from all starts
    for start in range(n):
        if time.time() > deadline - 5.0:
            break
            
        tour = _construct_nn_tour(distance_matrix, start)
        tour = _refine_tour(tour, distance_matrix, deadline)
        
        tour_len = _tour_length(tour, distance_matrix)
        if tour_len < best_len:
            best_len = tour_len
            best_tour = tour.copy()
            
    # 2. Greedy Insertion from a few starts (e.g., 0, 1, 2, or random)
    # Greedy Insertion is O(N^3) construction, so it's slower than NN.
    # Let's try starts 0 to min(5, n-1)
    gi_starts = list(range(min(5, n)))
    for start in gi_starts:
        if time.time() > deadline - 5.0:
            break
            
        tour = _construct_greedy_insertion_tour(distance_matrix, start)
        tour = _refine_tour(tour, distance_matrix, deadline)
        
        tour_len = _tour_length(tour, distance_matrix)
        if tour_len < best_len:
            best_len = tour_len
            best_tour = tour.copy()
            
    if best_tour is None:
        best_tour = np.arange(n)
        
    return best_tour

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _DEADLINE, _CACHE, _START_TIME
    
    if _START_TIME is None:
        _START_TIME = time.time()
        _DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill
        
    # Deadline guard: fall back to nearest neighbor if time is up
    if time.time() > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        d0 = distance_matrix[current_node, unvisited_nodes[0]]
        d1 = distance_matrix[current_node, unvisited_nodes[1]]
        d_dest_0 = distance_matrix[unvisited_nodes[0], destination_node]
        d_dest_1 = distance_matrix[unvisited_nodes[1], destination_node]
        d_cross = distance_matrix[unvisited_nodes[0], unvisited_nodes[1]]
        
        cost_pick_0 = d0 + d_cross + d_dest_1
        cost_pick_1 = d1 + d_cross + d_dest_0
        
        if cost_pick_0 <= cost_pick_1:
            return int(unvisited_nodes[0])
        else:
            return int(unvisited_nodes[1])

    instance_id = _get_instance_id(distance_matrix)
    if instance_id in _CACHE:
        full_tour = _CACHE[instance_id]
    else:
        # Construct the best possible tour within the time budget
        full_tour = _construct_best_tour(distance_matrix, _DEADLINE)
        _CACHE[instance_id] = full_tour
        
    try:
        idx = np.where(full_tour == current_node)[0][0]
        next_node = full_tour[(idx + 1) % len(full_tour)]
    except IndexError:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    if next_node in unvisited_nodes:
        return int(next_node)
    else:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

# EVOLVE-BLOCK-END
