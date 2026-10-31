"""TSP Constructive Heuristic: Reference Tour Following with Or-Opt Enhancement.

Strategy:
  1. Pre-compute a high-quality reference tour using Multi-Start Nearest-Neighbor,
     followed by 2-Opt and Or-Opt local search to convergence.
  2. During the constructive phase, strictly follow the successor of the current
     node in the reference tour if that successor is still unvisited.
  3. If the reference successor is already visited, fall back to a local greedy
     heuristic (Nearest Neighbor with destination penalty) or exact brute force
     for small remaining sets.
  4. This approach leverages the global quality of the 2-Opt/Or-Opt tour while
     maintaining the constructive step-by-step requirement.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import permutations

_start_time = time.time()
_LIMIT = 145.0  # Safety margin before 150s limit

# Per-instance state
_instance_key = None
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
    """Improve a tour using Or-Opt (moving segments of length 1, 2, or 3)."""
    n = len(tour)
    if n < 4:
        return tour
    
    tour = list(tour)
    improved = True
    
    while improved and not _is_deadline_passed():
        improved = False
        # Try moving segments of length 1, 2, 3
        for seg_len in [1, 2, 3]:
            if seg_len >= n:
                continue
            for i in range(n):
                if _is_deadline_passed():
                    return tour
                # Segment to move is tour[i : i + seg_len]
                # Predecessor of segment
                pred = tour[(i - 1) % n]
                # Successor of segment
                succ = tour[(i + seg_len) % n]
                
                # Cost before moving
                # Edges involved: (pred, tour[i]), (tour[i+seg_len-1], succ)
                cost_before = distance_matrix[pred][tour[i]] + distance_matrix[tour[i + seg_len - 1]][succ]
                
                # Try inserting the segment after each other node k
                # k cannot be i-1, i, ..., i+seg_len-1, i+seg_len
                for k in range(n):
                    if k >= i and k < i + seg_len + 1:
                        continue
                    # Insertion point: after k
                    # New edges: (tour[k], tour[i]), (tour[i+seg_len-1], tour[(k+1)%n])
                    # Note: if k is the predecessor of the segment in the new position, skip
                    # Actually, we are moving the segment to a new location.
                    # The segment is removed from i..i+seg_len-1.
                    # It is inserted after k.
                    # We must ensure we don't insert it back in the same place.
                    
                    # The node after k in the current tour (before removal) is tour[(k+1)%n]
                    # But if k is inside the segment being moved, this logic is tricky.
                    # Standard Or-Opt:
                    # Remove segment S from position i.
                    # Insert S after position k.
                    # k should not be in [i-1, i+seg_len-1] (indices mod n)
                    
                    # Let's simplify: just try all k not in the segment and not the immediate neighbors that result in no change.
                    
                    # Indices in current tour
                    seg_start = i
                    seg_end = (i + seg_len) % n # This is the node AFTER the segment
                    
                    # Nodes in segment: tour[i], ..., tour[(i+seg_len-1)%n]
                    # Predecessor: tour[(i-1)%n]
                    # Successor: tour[(i+seg_len)%n]
                    
                    # If we insert after k:
                    # New predecessor of segment: tour[k]
                    # New successor of segment: tour[(k+1)%n] (if k is not the last node before the segment in the original, but we are moving it)
                    
                    # Let's just calculate the delta directly.
                    # Original edges: (pred, start_seg), (end_seg, succ)
                    # New edges: (tour[k], start_seg), (end_seg, tour[(k+1)%n])
                    # We must ensure that the edge (tour[k], tour[(k+1)%n]) exists in the current tour and is broken.
                    # If k is such that tour[k] and tour[(k+1)%n] are not adjacent in the current tour, this is invalid.
                    # In a tour, every node is adjacent to the next. So (k, k+1) is always an edge.
                    
                    # However, if k is part of the segment being moved, we can't use it.
                    # Also, if k is the predecessor of the segment (i-1), inserting after k puts it back in place (roughly).
                    # If k is the node before the predecessor, etc.
                    
                    # Let's check if k is in the segment
                    in_segment = False
                    for m in range(seg_len):
                        if (i + m) % n == k:
                            in_segment = True
                            break
                    if in_segment:
                        continue
                    
                    # Also skip if k is the immediate predecessor or successor such that the move is trivial or invalid
                    # If k == (i-1)%n, we are inserting after pred, which is the same place.
                    # If k == (i+seg_len-1)%n, we are inserting after the last node of the segment, which is the same place.
                    
                    # Calculate new cost
                    new_pred = tour[k]
                    new_succ = tour[(k + 1) % n]
                    
                    # If the segment is inserted such that it overlaps with itself, skip.
                    # This is handled by the in_segment check.
                    
                    cost_after = distance_matrix[new_pred][tour[i]] + distance_matrix[tour[i + seg_len - 1]][new_succ]
                    
                    delta = cost_after - cost_before
                    if delta < -1e-10:
                        # Perform the move
                        # 1. Remove segment
                        seg = tour[i:i+seg_len]
                        tour = tour[:i] + tour[i+seg_len:]
                        # 2. Find new index for k in the modified tour
                        # Since we removed nodes, indices shift.
                        # It's safer to reconstruct the tour list explicitly.
                        # But for speed, let's just use the list slice method carefully.
                        
                        # Actually, list slicing is easier to reason about for correctness in a quick implementation.
                        # Let's re-implement the move using list operations.
                        # This is slower but n=50 is small.
                        
                        # Reset tour to original before move
                        tour = list(tour)
                        # We need the original tour to extract the segment
                        # The 'tour' variable was modified above? No, we didn't modify it yet.
                        # Wait, the code above `tour = tour[:i]...` was just a check.
                        # Let's do it properly.
                        
                        # Extract segment
                        seg_to_move = tour[i:i+seg_len]
                        
                        # Remove from original position
                        # Note: if i+seg_len > n, it wraps. But list slice doesn't wrap.
                        # So we must handle wrap-around.
                        # For simplicity and correctness with n=50, let's rotate the tour so the segment is at the start, or handle indices carefully.
                        # Given the low n, a simple list manipulation with wrap handling is fine.
                        
                        # Let's just use the standard Or-Opt move logic with list reversal/rotation if needed.
                        # For now, let's skip the complex wrap-around logic in the inner loop to avoid bugs, 
                        # and rely on 2-Opt which is already strong. Or-Opt is an enhancement.
                        # If we can't implement Or-Opt safely quickly, we can omit it.
                        # However, the cell finding said it helps.
                        
                        # Let's try a simpler Or-Opt: only move segments that don't wrap around the end of the list.
                        # i.e., i + seg_len <= n.
                        if i + seg_len <= n:
                            # No wrap
                            # Remove
                            new_tour = tour[:i] + tour[i+seg_len:]
                            # Find index of k in new_tour
                            try:
                                k_idx = new_tour.index(k)
                            except ValueError:
                                continue
                            # Insert after k_idx
                            # The node after k in new_tour is new_tour[k_idx+1]
                            # We insert seg_to_move at k_idx+1
                            new_tour = new_tour[:k_idx+1] + seg_to_move + new_tour[k_idx+1:]
                            # Verify length
                            if len(new_tour) != n:
                                continue
                            tour = new_tour
                            improved = True
                            break # Restart outer loops? Or just continue?
                        # If it wraps, we skip for now to ensure safety.
                if improved:
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
        tour_or_opt = _or_opt(distance_matrix, tour_2opt)
        
        # Compute tour length
        tour_len = sum(distance_matrix[tour_or_opt[i]][tour_or_opt[(i + 1) % n]] for i in range(n))
        
        if tour_len < best_len:
            best_len = tour_len
            best_tour = tour_or_opt
    
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
    global _instance_key
    
    n_total = distance_matrix.shape[0]
    n_unvisited = len(unvisited_nodes)
    
    # A new instance starts when all (or nearly all) nodes are unvisited
    if n_unvisited >= n_total - 1:
        # Quick checksum
        step = max(1, n_total // 10)
        checksum = np.sum(distance_matrix[::step, ::step])
        
        if _instance_key is None or abs(checksum - _instance_key) > 1e-6:
            _instance_key = checksum
            return True
    
    return False


def _brute_force_tail(distance_matrix, current_node, unvisited_nodes, destination_node):
    """
    Solve the tail of the tour exactly for small sets.
    Returns the first node in the optimal permutation.
    """
    nodes = list(unvisited_nodes)
    n = len(nodes)
    
    if n == 0:
        return -1
    if n == 1:
        return nodes[0]
    
    best_cost = float('inf')
    best_first = nodes[0]
    
    # Fix the first node? No, we need to pick the first node from unvisited.
    # We are at current_node. We must visit all nodes in 'nodes' and end at destination_node.
    # So we are looking for a permutation of 'nodes' that minimizes:
    # dist(current, p[0]) + dist(p[0], p[1]) + ... + dist(p[-1], destination)
    
    for p in permutations(nodes):
        cost = distance_matrix[current_node][p[0]]
        for i in range(len(p) - 1):
            cost += distance_matrix[p[i]][p[i+1]]
        cost += distance_matrix[p[-1]][destination_node]
        
        if cost < best_cost:
            best_cost = cost
            best_first = p[0]
            
    return int(best_first)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Follows the pre-computed reference tour's successor when possible,
    falling back to local scoring or brute force when the reference successor is unavailable.
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
        # Using a set for O(1) lookup if n is large, but n=50 is small.
        if ref_next in unvisited_nodes:
            return ref_next
    
    # Fallback: if reference successor is visited, use local heuristic
    # If the number of unvisited nodes is small (<= 5), use brute force
    if n_unvisited <= 5:
        return _brute_force_tail(distance_matrix, current_node, unvisited_nodes, destination_node)
    
    # Otherwise, use greedy: minimize dist(current, next) + 0.1 * dist(next, destination)
    # This helps guide the path back to the destination if we've deviated from the reference tour.
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    scores = dists_from_current + 0.1 * dists_to_dest
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
