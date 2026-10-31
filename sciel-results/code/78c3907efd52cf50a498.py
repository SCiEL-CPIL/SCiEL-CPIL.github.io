"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import hashlib

# Global state
_DEADLINE = None
_CACHE = {}
_START_TIME = None

def _get_instance_id(distance_matrix):
    """Create a lightweight fingerprint for the distance matrix."""
    n = distance_matrix.shape[0]
    if n > 10:
        # Use a larger sample for better uniqueness, but keep it fast
        step = max(1, n // 20)
        sample = distance_matrix[::step, ::step].tobytes()
    else:
        sample = distance_matrix.tobytes()
    # Use a stable hash
    return hashlib.md5(sample).digest()

def _two_opt_cycle(tour, distance_matrix, deadline):
    """
    Apply 2-Opt to a full TSP cycle with efficient scanning.
    Uses a more aggressive search with multiple passes and restarts.
    """
    n = len(tour)
    if n <= 3:
        return tour
        
    tour = tour.copy()
    
    def get_cost(t):
        return np.sum(distance_matrix[t[:-1], t[1:]]) + distance_matrix[t[-1], t[0]]
    
    current_cost = get_cost(tour)
    
    improved = True
    passes = 0
    max_passes = 200
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        
        if time.time() > deadline - 1.0:
            break
            
        # First pass: standard 2-opt
        for i in range(n - 1):
            if time.time() > deadline - 1.0:
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
                    break
        
    return tour

def _or_opt_cycle(tour, distance_matrix, deadline):
    """
    Apply Or-Opt moves to a full TSP cycle.
    Moves segments of length 1, 2, or 3 to other positions.
    """
    n = len(tour)
    if n <= 4:
        return tour
        
    tour = tour.copy()
    
    def get_cost(t):
        return np.sum(distance_matrix[t[:-1], t[1:]]) + distance_matrix[t[-1], t[0]]
    
    current_cost = get_cost(tour)
    
    improved = True
    passes = 0
    max_passes = 50
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        
        if time.time() > deadline - 1.0:
            break
            
        for seg_len in [1, 2, 3]:
            if time.time() > deadline - 1.0:
                break
                
            for i in range(n - seg_len):
                if time.time() > deadline - 1.0:
                    break
                    
                # Segment to move: tour[i : i+seg_len]
                seg = tour[i : i+seg_len]
                
                # Nodes before and after the segment
                before = tour[i-1] if i > 0 else tour[n-1]
                after = tour[i+seg_len] if i+seg_len < n else tour[0]
                
                # Cost of edges around the segment (to be removed)
                d_before_seg = distance_matrix[before, seg[0]]
                d_seg_after = distance_matrix[seg[-1], after]
                
                # Try inserting the segment at each other position
                for j in range(n - seg_len):
                    if j == i or (j > i and j < i + seg_len):
                        continue
                        
                    # Position j: insert before tour[j]
                    new_before = tour[j-1] if j > 0 else tour[n-1]
                    new_after = tour[j]
                    
                    d_new_before_seg = distance_matrix[new_before, seg[0]]
                    d_new_seg_after = distance_matrix[seg[-1], new_after]
                    
                    delta = d_new_before_seg + d_new_seg_after - d_before_seg - d_seg_after
                    
                    if delta < -1e-10:
                        # Remove segment from position i
                        tour = np.concatenate([tour[:i], tour[i+seg_len:]])
                        n = len(tour)
                        
                        # Adjust j if it was after i
                        if j >= i:
                            j = j - seg_len
                        if j < 0:
                            j = n + j
                        
                        # Insert segment at position j
                        if j == 0:
                            tour = np.concatenate([seg, tour])
                        else:
                            tour = np.concatenate([tour[:j], seg, tour[j:]])
                        
                        current_cost += delta
                        improved = True
                        break
                
                if improved:
                    break
            
            if improved:
                break
                
    return tour

def _construct_tour(distance_matrix, deadline):
    """
    Construct a high-quality tour using Multi-Start Nearest Neighbor + 2-Opt + Or-Opt.
    Returns the best tour found as a numpy array of node indices.
    """
    n = distance_matrix.shape[0]
    best_tour = None
    best_len = np.inf
    
    # Use more starting points for better quality
    for start in range(n):
        if time.time() > deadline - 5.0:
            break
            
        # Construct NN tour starting from 'start'
        remaining = set(range(n))
        remaining.remove(start)
        tour = [start]
        curr = start
        
        while remaining:
            dists = distance_matrix[curr, list(remaining)]
            min_idx = np.argmin(dists)
            next_node = list(remaining)[min_idx]
            tour.append(next_node)
            remaining.remove(next_node)
            curr = next_node
        
        tour_arr = np.array(tour)
        
        # Apply 2-Opt
        tour_arr = _two_opt_cycle(tour_arr, distance_matrix, deadline)
        
        # Apply Or-Opt
        tour_arr = _or_opt_cycle(tour_arr, distance_matrix, deadline)
        
        # Final 2-Opt pass after Or-Opt
        tour_arr = _two_opt_cycle(tour_arr, distance_matrix, deadline)
        
        tour_len = np.sum(distance_matrix[tour_arr[:-1], tour_arr[1:]]) + distance_matrix[tour_arr[-1], tour_arr[0]]
        
        if tour_len < best_len:
            best_len = tour_len
            best_tour = tour_arr
            
        if time.time() > deadline - 5.0:
            break
            
    if best_tour is None:
        best_tour = np.arange(n)
        
    return best_tour

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _DEADLINE, _CACHE, _START_TIME
    
    if _START_TIME is None:
        _START_TIME = time.time()
        _DEADLINE = _START_TIME + 145.0  # 5s safety margin before 150s kill
        
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
        full_tour = _construct_tour(distance_matrix, _DEADLINE)
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
