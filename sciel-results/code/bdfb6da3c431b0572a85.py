"""
Reference-Tour Guided TSP Constructive Heuristic with 2-Opt and Or-Opt.

Strategy:
  1. On detecting a new instance, build a high-quality reference tour using
     Multi-Start Nearest-Neighbor (all starts) followed by 2-Opt and Or-Opt 
     local search to convergence.
  2. Pre-compute the successor for each node in that reference tour.
  3. During construction, follow the reference tour's successor when possible.
     If the successor is not available (already visited), fall back to a 
     nearest-neighbor choice among unvisited nodes.
  4. This approach leverages the global optimization of 2-Opt/Or-Opt without
     needing to solve the TSP online, providing a strong baseline for the 
     constructive phase.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 115.0  # Safety margin before 120s limit

# Per-instance state
_last_dm_checksum = None
_reference_successor = None  # array: reference_successor[i] = next node after i in ref tour
_instance_initialized = False


def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT


def _nn_tour(distance_matrix, start):
    """Build a tour using nearest neighbor from a given start node."""
    n = distance_matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = [start]
    current = start
    
    for _ in range(n - 1):
        # Find nearest unvisited
        dists = distance_matrix[current].copy()
        dists[visited] = np.inf
        next_node = int(np.argmin(dists))
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    
    return tour


def _two_opt(distance_matrix, tour):
    """Improve a tour using 2-opt until no improvement is found."""
    n = len(tour)
    if n < 4:
        return tour
    
    tour = list(tour)
    improved = True
    
    while improved and not _is_deadline_passed():
        improved = False
        for i in range(n - 2):
            if _is_deadline_passed():
                return tour
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Calculate delta
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                delta = distance_matrix[a][c] + distance_matrix[b][d] - distance_matrix[a][b] - distance_matrix[c][d]
                if delta < -1e-10:
                    # Reverse segment
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    return tour


def _or_opt(distance_matrix, tour):
    """Improve a tour using Or-Opt (relocate 1, 2, or 3 node segments)."""
    n = len(tour)
    if n < 4:
        return tour
    
    tour = list(tour)
    improved = True
    
    while improved and not _is_deadline_passed():
        improved = False
        
        # Relocate 1 node
        for i in range(n):
            if _is_deadline_passed():
                return tour
            node = tour[i]
            prev = tour[(i - 1) % n]
            next_node = tour[(i + 1) % n]
            
            # Try inserting node between j and j+1
            for j in range(n):
                if j == i or j == (i - 1) % n or j == (i + 1) % n:
                    continue
                if j == (i + n - 1) % n: # j is prev
                    continue
                if j == (i + 1) % n: # j is next
                    continue
                
                j_next = tour[(j + 1) % n]
                
                # Cost change:
                # Remove i: -d(prev, i) - d(i, next_node)
                # Insert i between j and j_next: +d(j, i) + d(i, j_next) - d(j, j_next)
                delta = (distance_matrix[j][node] + distance_matrix[node][j_next] - distance_matrix[j][j_next]) \
                        - (distance_matrix[prev][node] + distance_matrix[node][next_node])
                
                if delta < -1e-10:
                    # Perform move
                    tour.pop(i)
                    # Find new index of j after pop
                    # Since we popped i, indices after i shift left by 1.
                    # If j > i, new_j = j - 1. If j < i, new_j = j.
                    # But it's safer to find j in the list.
                    # For n=50, linear search is fine, but let's be efficient.
                    if j > i:
                        insert_pos = j
                    else:
                        insert_pos = j + 1 # because we popped i, j's index is unchanged if j<i, 
                                           # but we are inserting AFTER j. 
                                           # Wait, if we pop i, the list shrinks.
                                           # Let's just find the index of j in the new list.
                        insert_pos = j
                    
                    # Actually, simpler: remove node, then find index of j, insert at index+1
                    new_tour = list(tour)
                    idx_j = new_tour.index(j)
                    new_tour.insert(idx_j + 1, node)
                    tour = new_tour
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue

        # Relocate 2 nodes
        for i in range(n):
            if _is_deadline_passed():
                return tour
            n1 = tour[i]
            n2 = tour[(i + 1) % n]
            prev = tour[(i - 1) % n]
            next_node = tour[(i + 2) % n]
            
            for j in range(n):
                if j in [i, (i + 1) % n, (i - 1) % n, (i + 2) % n]:
                    continue
                j_next = tour[(j + 1) % n]
                
                # Cost change:
                # Remove i, i+1: -d(prev, n1) - d(n1, n2) - d(n2, next_node)
                # Insert between j, j_next: +d(j, n1) + d(n1, n2) + d(n2, j_next) - d(j, j_next)
                # Note: d(n1, n2) cancels out if we keep them together.
                delta = (distance_matrix[j][n1] + distance_matrix[n2][j_next] - distance_matrix[j][j_next]) \
                        - (distance_matrix[prev][n1] + distance_matrix[n2][next_node])
                
                if delta < -1e-10:
                    # Perform move
                    # Remove n1, n2
                    tour.remove(n1)
                    tour.remove(n2)
                    idx_j = tour.index(j)
                    tour.insert(idx_j + 1, n2)
                    tour.insert(idx_j + 1, n1)
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue

        # Relocate 3 nodes
        for i in range(n):
            if _is_deadline_passed():
                return tour
            n1 = tour[i]
            n2 = tour[(i + 1) % n]
            n3 = tour[(i + 2) % n]
            prev = tour[(i - 1) % n]
            next_node = tour[(i + 3) % n]
            
            for j in range(n):
                if j in [i, (i + 1) % n, (i + 2) % n, (i - 1) % n, (i + 3) % n]:
                    continue
                j_next = tour[(j + 1) % n]
                
                # Cost change:
                # Remove i, i+1, i+2: -d(prev, n1) - d(n1, n2) - d(n2, n3) - d(n3, next_node)
                # Insert between j, j_next: +d(j, n1) + d(n1, n2) + d(n2, n3) + d(n3, j_next) - d(j, j_next)
                # d(n1, n2) and d(n2, n3) cancel out.
                delta = (distance_matrix[j][n1] + distance_matrix[n3][j_next] - distance_matrix[j][j_next]) \
                        - (distance_matrix[prev][n1] + distance_matrix[n3][next_node])
                
                if delta < -1e-10:
                    # Perform move
                    tour.remove(n1)
                    tour.remove(n2)
                    tour.remove(n3)
                    idx_j = tour.index(j)
                    tour.insert(idx_j + 1, n3)
                    tour.insert(idx_j + 1, n2)
                    tour.insert(idx_j + 1, n1)
                    improved = True
                    break
            if improved:
                break

    return tour


def _build_reference_tour(distance_matrix):
    """
    Build a high-quality reference tour using Multi-Start NN + 2-Opt + Or-Opt.
    Returns an array where reference_successor[i] is the node visited after i in the tour.
    """
    global _reference_successor
    
    n = distance_matrix.shape[0]
    if n <= 1:
        return np.array([], dtype=int)
    if n == 2:
        return np.array([1, 0])
    
    best_tour = None
    best_len = float('inf')
    
    # Multi-start: try all starting nodes
    for start in range(n):
        if _is_deadline_passed():
            break
        
        # Build NN tour from this start
        tour = _nn_tour(distance_matrix, start)
        
        # Apply 2-opt
        tour_2opt = _two_opt(distance_matrix, tour)
        
        # Apply Or-Opt
        tour_oropt = _or_opt(distance_matrix, tour_2opt)
        
        # Apply 2-opt again after Or-Opt to clean up
        tour_final = _two_opt(distance_matrix, tour_oropt)
        
        # Compute tour length
        tour_len = sum(distance_matrix[tour_final[i]][tour_final[(i + 1) % n]] for i in range(n))
        
        if tour_len < best_len:
            best_len = tour_len
            best_tour = tour_final
    
    if best_tour is None:
        # Fallback: simple NN from node 0
        best_tour = _nn_tour(distance_matrix, 0)
    
    # Build successor array
    successor = np.zeros(n, dtype=int)
    for i in range(n):
        successor[best_tour[i]] = best_tour[(i + 1) % n]
    
    _reference_successor = successor
    return successor


def _detect_new_instance(distance_matrix, unvisited_nodes):
    """Detect if we're starting a new instance."""
    global _last_dm_checksum
    
    n_total = distance_matrix.shape[0]
    n_unvisited = len(unvisited_nodes)
    
    # A new instance starts when all (or nearly all) nodes are unvisited
    if n_unvisited >= n_total - 1:
        # Quick checksum
        step = max(1, n_total // 10)
        checksum = np.sum(distance_matrix[::step, ::step])
        
        if _last_dm_checksum is None or abs(checksum - _last_dm_checksum) > 1e-6:
            _last_dm_checksum = checksum
            return True
    
    return False


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Follows the pre-computed reference tour's successor when possible,
    falling back to nearest neighbor when the reference successor is unavailable.
    """
    global _instance_initialized, _reference_successor
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Detect new instance
    is_new_instance = _detect_new_instance(distance_matrix, unvisited_nodes)
    
    if is_new_instance:
        _instance_initialized = False
        _reference_successor = None
        # Build reference tour
        _build_reference_tour(distance_matrix)
        _instance_initialized = True
    
    if _is_deadline_passed():
        # Fallback: pure nearest neighbor
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])
    
    # Try to follow the reference tour
    if _reference_successor is not None:
        ref_next = int(_reference_successor[current_node])
        # Check if ref_next is in unvisited_nodes
        # Using 'in' on a list is O(N). For N=50, this is fine.
        # If we want to be faster, we could maintain a set of unvisited nodes,
        # but we don't have state access to that here easily without global vars.
        # Given N=50, O(N) is negligible.
        if ref_next in unvisited_nodes:
            return ref_next
    
    # Fallback: local scoring with small lookahead
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    scores = dists_from_current.copy()
    
    # Add a small penalty for nodes that are far from the destination
    if n_unvisited > 2:
        dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
        scores += 0.1 * dists_to_dest
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
