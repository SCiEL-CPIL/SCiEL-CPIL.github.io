"""Traveling salesman constructive heuristic with dynamic partial-path optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0

# Cache for pre-computed global tours (fallback or initialization)
_tour_cache = {}

def _get_dm_fingerprint(dm):
    """Create a cheap, unique fingerprint of the distance matrix."""
    return float(np.sum(dm[:5, :5]) + np.sum(dm[-5:, -5:]) + np.sum(dm[10:15, 10:15]))

def _nearest_neighbor_tour(n, dist_matrix, start=0):
    """Generate a tour using the Nearest Neighbor heuristic starting from node start."""
    visited = [False] * n
    tour = [start]
    visited[start] = True
    current = start
    
    for _ in range(n - 1):
        min_dist = float('inf')
        next_node = -1
        # Vectorized NN step
        dists = dist_matrix[current, :]
        dists[visited] = float('inf')
        next_node = int(np.argmin(dists))
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
        
    return tour

def _two_opt_fast(tour, dist_matrix, time_limit):
    """
    Fast 2-Opt improvement.
    tour: list of nodes
    dist_matrix: numpy array
    time_limit: seconds
    Returns improved tour.
    """
    n = len(tour)
    if n < 4:
        return tour
    
    tour = list(tour)
    t_start = time.time()
    improved = True
    
    # Pre-compute edge lengths for speed? No, direct indexing is fast enough for N=50.
    
    while improved and (time.time() - t_start < time_limit):
        improved = False
        for i in range(n - 1):
            if time.time() - t_start > time_limit:
                return tour
            a = tour[i]
            b = tour[i + 1]
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                c = tour[j]
                d = tour[(j + 1) % n]
                
                # Delta = d(a,c) + d(b,d) - d(a,b) - d(c,d)
                delta = (dist_matrix[a, c] + dist_matrix[b, d]) - (dist_matrix[a, b] + dist_matrix[c, d])
                
                if delta < -1e-10:
                    # Reverse segment i+1 to j
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    improved = True
                    
    return tour

def _or_opt_fast(tour, dist_matrix, time_limit):
    """
    Fast Or-Opt improvement (segment sizes 1, 2, 3).
    """
    n = len(tour)
    if n < 4:
        return tour
    
    tour = list(tour)
    t_start = time.time()
    improved = True
    
    while improved and (time.time() - t_start < time_limit):
        improved = False
        for seg_len in [1, 2, 3]:
            if time.time() - t_start > time_limit:
                return tour
            for i in range(n):
                if time.time() - t_start > time_limit:
                    return tour
                
                prev_i = (i - 1) % n
                next_i = (i + seg_len) % n
                
                # Extract segment
                segment = [tour[(i + k) % n] for k in range(seg_len)]
                first_node = segment[0]
                last_node = segment[-1]
                
                # Cost of removing segment
                # Remove edges: (prev_i, first_node), (last_node, next_i), (j, j+1)
                # Add edges: (prev_i, next_i), (j, first_node), (last_node, j+1)
                
                c_old1 = dist_matrix[tour[prev_i], first_node]
                c_old2 = dist_matrix[last_node, tour[next_i]]
                c_new_base = dist_matrix[tour[prev_i], tour[next_i]]
                
                for j in range(n):
                    if time.time() - t_start > time_limit:
                        return tour
                    
                    next_j = (j + 1) % n
                    
                    # Skip if insertion point overlaps with segment
                    # Segment indices in tour: i, i+1, ..., i+seg_len-1
                    # We are iterating j over 0..n-1
                    # If j is in the segment range, skip.
                    # Note: modular arithmetic makes this tricky. 
                    # Simple check: if the node at j or j+1 is in the segment, skip.
                    # Actually, if we remove the segment, the indices shift.
                    # It's safer to check node identity.
                    seg_set = set(segment)
                    if tour[j] in seg_set or tour[next_j] in seg_set:
                        continue
                    
                    # Also skip trivial moves (inserting back into same gap)
                    if j == prev_i or j == (i - 2) % n:
                        # If we insert between prev_i and next_i, that's the original spot (roughly)
                        # Specifically, if j == prev_i, we insert between prev_i and next_i.
                        # This is the original position.
                        # If j == (i-2)%n, we insert between i-2 and i-1? No, 
                        # Let's just rely on delta > 0 check. Trivial moves have delta 0.
                        pass

                    c_old3 = dist_matrix[tour[j], tour[next_j]]
                    c_new2 = dist_matrix[tour[j], first_node]
                    c_new3 = dist_matrix[last_node, tour[next_j]]
                    
                    delta = (c_new_base + c_new2 + c_new3) - (c_old1 + c_old2 + c_old3)
                    
                    if delta < -1e-10:
                        # Perform move
                        new_tour = tour.copy()
                        # Remove segment
                        del new_tour[i:i+seg_len]
                        
                        # Adjust j for insertion
                        # If j > i, j shifts left by seg_len
                        # If j < i, j stays
                        # But we must be careful with wrap-around if i is near end?
                        # Since we treat tour as a list 0..n-1, and segment is at i..i+L-1:
                        # If j >= i + seg_len, new_j = j - seg_len
                        # If j < i, new_j = j
                        # We already skipped j in segment.
                        
                        if j >= i + seg_len:
                            insert_idx = j - seg_len
                        else:
                            insert_idx = j
                        
                        new_tour[insert_idx:insert_idx] = segment
                        tour = new_tour
                        improved = True
                        
    return tour

def _estimate_tour_length(tour, dist_matrix, unvisited_nodes, current_node):
    """
    Estimate the final tour length given a partial path `tour` (ending at current_node)
    and a set of `unvisited_nodes`.
    We complete the path greedily (NN) from current_node through unvisited_nodes back to start(tour[0]).
    """
    if not unvisited_nodes:
        # Closed tour
        return sum(dist_matrix[tour[i], tour[(i+1)%len(tour)]] for i in range(len(tour)))
    
    # Greedy completion
    visited = set(tour)
    curr = current_node
    length = 0.0
    
    # Add edges from current partial path? No, we just need the completion cost.
    # The partial path length is fixed. We need to add:
    # 1. Path from current_node through all unvisited back to tour[0]
    
    remaining = list(unvisited_nodes)
    remaining_set = set(remaining)
    
    # Start from current_node
    # We need to visit all in remaining and return to tour[0]
    
    # NN heuristic for the completion
    curr = current_node
    while remaining:
        # Find nearest in remaining
        dists = dist_matrix[curr, :]
        # Mask visited
        # For speed, since N=50, simple loop is fine, or argmin on masked array
        min_d = float('inf')
        next_node = -1
        for node in remaining:
            d = dist_matrix[curr, node]
            if d < min_d:
                min_d = d
                next_node = node
        length += min_d
        remaining.remove(next_node)
        curr = next_node
    
    # Return to start of original tour
    length += dist_matrix[curr, tour[0]]
    
    return length

def _improve_partial_path(tour, dist_matrix, unvisited_nodes, time_budget):
    """
    Improve the partial path `tour` (which ends at the last element) by applying
    2-Opt and Or-Opt to the partial path, guided by the estimated final tour length.
    
    Note: Standard 2-Opt minimizes path length. For a partial path, minimizing path length
    is a good proxy, but minimizing the *estimated final tour length* is better.
    However, 2-Opt delta calculation is local. 
    For simplicity and speed, we will apply standard 2-Opt to the partial path.
    This is valid because a shorter partial path usually leads to a shorter final tour,
    assuming the completion strategy is consistent.
    """
    if len(tour) < 3:
        return tour
    
    # Apply 2-Opt to the partial path
    # We need to be careful: 2-Opt reverses segments.
    # Does this preserve the "start" and "end"? 
    # Yes, if we treat the partial path as an open chain, 2-Opt on the chain
    # (excluding the closing edge) is valid.
    
    t_start = time.time()
    
    # 2-Opt on open path
    n = len(tour)
    improved = True
    while improved and (time.time() - t_start < time_budget * 0.5):
        improved = False
        for i in range(n - 1):
            if time.time() - t_start > time_budget * 0.5:
                break
            a = tour[i]
            b = tour[i + 1]
            for j in range(i + 2, n):
                c = tour[j]
                d = tour[(j + 1) % n] # Careful: if j is last, d is tour[0]? 
                # In an open path, we don't have an edge from last to first.
                # So 2-Opt on open path:
                # Edges: (i, i+1) and (j, j+1) are removed?
                # No, standard 2-Opt on a path:
                # Remove (a,b) and (c,d), add (a,c) and (b,d).
                # Here b=tour[i+1], a=tour[i].
                # c=tour[j], d=tour[j+1].
                # If j == n-1, d is undefined in the path context? 
                # Actually, the path is 0->1->...->n-1.
                # Edges are (0,1), (1,2), ..., (n-2, n-1).
                # 2-Opt move on edges (i, i+1) and (j, j+1) where i < j.
                # If j = n-1, there is no edge (n-1, n). 
                # So j must be <= n-2.
                
                if j >= n - 1:
                    continue
                    
                c = tour[j]
                d = tour[j + 1]
                
                delta = (dist_matrix[a, c] + dist_matrix[b, d]) - (dist_matrix[a, b] + dist_matrix[c, d])
                
                if delta < -1e-10:
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    improved = True
    
    # Or-Opt on open path (simplified: just size 1)
    t_start2 = time.time()
    improved = True
    while improved and (time.time() - t_start2 < time_budget * 0.5):
        improved = False
        for i in range(1, n - 1): # Don't move start or end
            if time.time() - t_start2 > time_budget * 0.5:
                break
            node = tour[i]
            prev = tour[i - 1]
            next_ = tour[i + 1]
            
            c_old1 = dist_matrix[prev, node]
            c_old2 = dist_matrix[node, next_]
            c_new_base = dist_matrix[prev, next_]
            
            for j in range(n):
                if j == i or j == i - 1 or j == i + 1:
                    continue
                # Insert between tour[j] and tour[j+1]
                # If j == n-1, insert after end? No, open path.
                # We can insert at any position 0..n-1 (before index j).
                # Let's say we remove node at i.
                # Gap is between prev and next_.
                # We insert node between tour[j] and tour[j+1].
                # If j == n-1, we append to end.
                
                if j == n - 1:
                    # Insert at end
                    # Remove node from i
                    # New edges: (prev, next_), (tour[n-1], node)
                    # Old edges: (prev, node), (node, next_), (tour[n-1], ?) 
                    # Wait, if we append, the last edge changes?
                    # Original: ... -> prev -> node -> next_ -> ... -> last
                    # New: ... -> prev -> next_ -> ... -> last -> node
                    # Edges removed: (prev, node), (node, next_)
                    # Edges added: (prev, next_), (last, node)
                    # But what about the edge before last? It stays.
                    # So delta = d(prev, next_) + d(last, node) - d(prev, node) - d(node, next_)
                    
                    last_node = tour[n - 1]
                    delta = (c_new_base + dist_matrix[last_node, node]) - (c_old1 + c_old2)
                    
                    if delta < -1e-10:
                        # Perform move
                        tour.pop(i)
                        tour.append(node)
                        improved = True
                        break # Restart loop as indices changed
                else:
                    # Insert between j and j+1
                    # If j < i, indices shift? 
                    # Let's just construct new tour to be safe, N is small.
                    new_tour = tour.copy()
                    new_tour.pop(i)
                    
                    # Adjust j
                    if j > i:
                        insert_idx = j - 1
                    else:
                        insert_idx = j
                        
                    new_tour[insert_idx:insert_idx] = [node]
                    
                    # Calculate length difference
                    # Old: ... prev-node-next ...
                    # New: ... prev-next ... j-node-j+1 ...
                    
                    # We already have c_old1, c_old2, c_new_base
                    # Need d(j, node) + d(node, j+1) - d(j, j+1)
                    
                    c_old3 = dist_matrix[tour[j], tour[j + 1]]
                    c_new2 = dist_matrix[tour[j], node]
                    c_new3 = dist_matrix[node, tour[j + 1]]
                    
                    delta = (c_new_base + c_new2 + c_new3) - (c_old1 + c_old2 + c_old3)
                    
                    if delta < -1e-10:
                        tour = new_tour
                        improved = True
                        break
                        
    return tour

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node by dynamically optimizing the partial path.
    """
    if len(unvisited_nodes) == 0:
        return -1
        
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Check time
    if time.time() > _DEADLINE:
        # Fallback to nearest neighbor
        dists = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists)
        return int(unvisited_nodes[best_idx])
    
    # We don't have the full history of the path in the function signature.
    # The function only gets `current_node`.
    # This is a CRITICAL constraint.
    # "select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix)"
    # We do NOT receive the list of previously visited nodes!
    
    # This means we CANNOT do dynamic partial-path optimization on the actual path
    # unless we cache the path state globally.
    
    # Let's check if we can reconstruct the path or if we must assume a specific path.
    # The evaluator builds the tour step-by-step.
    # If we don't know the previous nodes, we can't optimize the partial path.
    
    # However, we CAN cache the state.
    # Key: (current_node, tuple(sorted(unvisited_nodes)))? 
    # No, unvisited_nodes is an np.ndarray, order might matter? 
    # Usually, the set of unvisited nodes determines the state, plus the current node.
    # But the *path* taken to get here matters for the tour length.
    
    # If we don't have the path, we are forced back to the "Pre-computed Tour" strategy
    # OR we must assume the evaluator follows a deterministic path that we can predict?
    # No, the evaluator calls our function to decide the next step.
    
    # Wait, the task says: "picks the next city while a traveling-salesman tour is built step by step".
    # If we don't have the history, we are in a "Markov" decision process where state is (current, unvisited).
    # But TSP is path-dependent.
    
    # Let's look at the inspirations. They all use "Pre-computed tour".
    # Inspiration 1: "pre-computes a near-optimal reference tour; following its successor"
    # This implies that the evaluator's path *matches* the pre-computed tour?
    # If the evaluator follows our advice, and we give it a pre-computed tour,
    # then the path IS the pre-computed tour.
    
    # So, if we return the next node from a pre-computed tour, the path is consistent with that tour.
    # This allows us to use the pre-computed tour as the "partial path" history.
    
    # Therefore, the "Dynamic Partial-Path Optimization" strategy can be implemented as:
    # 1. Maintain a cached "best known tour" for this instance.
    # 2. On each call, verify that `current_node` is in this tour.
    # 3. If yes, return the next node in this tour.
    # 4. If no (or if we want to improve), we can't really "fix" the path retroactively.
    
    # So the "Dynamic" part must be in *how we compute the initial tour* or *if we allow deviations*.
    
    # Alternative interpretation:
    # Maybe we can use the "estimated final tour length" to choose between multiple candidates
    # if we were doing a beam search? But we only return one node.
    
    # Given the constraint of no history, the most robust strategy remains:
    # 1. Compute a high-quality tour once (using NN + 2-Opt + Or-Opt + maybe ILS).
    # 2. Follow it.
    
    # To improve on the current program (which does NN + 2-Opt + Or-Opt),
    # we can use a stronger local search: **Lin-Kernighan (L-K)** or **Lin-Kernighan-Helsgaun (LKH)**-like moves,
    # or simply **more aggressive 2-Opt/Or-Opt** with a longer time budget,
    # or **Multi-Start** to ensure we find a better local optimum.
    
    # Let's try **Multi-Start Nearest Neighbor + 2-Opt**.
    # Start NN from multiple different starting nodes, optimize each, keep the best.
    # This is a structural change from "Single Start NN".
    
    # Also, we can use **Greedy Insertion** instead of NN for initial construction,
    # which often yields better initial tours for 2-Opt to refine.
    
    # Strategy:
    # 1. On first call, compute best tour using:
    #    a. Greedy Insertion (Farthest/Nearest insertion)
    #    b. 2-Opt
    #    c. Or-Opt
    #    d. Repeat with multiple random restarts if time permits.
    # 2. Cache the best tour.
    # 3. Follow the cached tour.
    
    # Let's implement Greedy Insertion.
    
    pass # Placeholder for actual logic below

# --- Actual Implementation ---

# Re-initialize cache and time
_tour_cache = {}
_START_TIME = time.time()
_DEADLINE = 145.0

def _get_dm_fingerprint(dm):
    """Create a cheap, unique fingerprint of the distance matrix."""
    # Use a hash of the first few elements and shape
    # For 50x50, summing corners is decent, but let's be safer with a simple hash of flattened array
    # Actually, summing is fast. Let's stick to it but add more points.
    return float(np.sum(dm[:5, :5]) + np.sum(dm[-5:, -5:]) + np.sum(dm[10:15, 10:15]) + np.sum(dm[20:25, 20:25]))

def _greedy_insertion_tour(n, dist_matrix, start=0):
    """
    Construct a tour using Greedy Insertion.
    Start with a triangle (start, nearest1, nearest2).
    Then insert the city that minimizes the increase in tour length.
    """
    # Initialize with 3 nodes
    visited = [False] * n
    tour = [start]
    visited[start] = True
    
    # Find 2 nearest neighbors of start
    dists = dist_matrix[start, :].copy()
    dists[start] = float('inf')
    n1 = int(np.argmin(dists))
    dists[n1] = float('inf')
    n2 = int(np.argmin(dists))
    
    tour.append(n1)
    tour.append(n2)
    visited[n1] = True
    visited[n2] = True
    
    unvisited = [i for i in range(n) if not visited[i]]
    
    while unvisited:
        best_cost = float('inf')
        best_node = -1
        best_pos = -1
        
        for node in unvisited:
            # Try inserting node at each position in current tour
            # Current tour is a cycle: tour[0] -> ... -> tour[-1] -> tour[0]
            # Positions are 1..len(tour) (insert after index i-1)
            # Or 0..len(tour)-1 (insert before index i)
            # Let's say we insert between tour[i-1] and tour[i].
            
            for i in range(len(tour)):
                prev_node = tour[(i - 1) % len(tour)]
                next_node = tour[i]
                
                # Cost increase = d(prev, node) + d(node, next) - d(prev, next)
                delta = dist_matrix[prev_node, node] + dist_matrix[node, next_node] - dist_matrix[prev_node, next_node]
                
                if delta < best_cost:
                    best_cost = delta
                    best_node = node
                    best_pos = i
        
        if best_node != -1:
            tour.insert(best_pos, best_node)
            visited[best_node] = True
            unvisited.remove(best_node)
        else:
            # Should not happen
            break
            
    return tour

def _two_opt(tour, dist_matrix, time_limit):
    """Improve the tour using 2-Opt moves until time limit is reached."""
    n = len(tour)
    if n < 4:
        return tour
    
    tour = list(tour)
    t_start = time.time()
    improved = True
    
    while improved and (time.time() - t_start < time_limit):
        improved = False
        for i in range(n - 1):
            if time.time() - t_start > time_limit:
                return tour
            a = tour[i]
            b = tour[i + 1]
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                
                c = tour[j]
                d = tour[(j + 1) % n]
                
                delta = (dist_matrix[a, c] + dist_matrix[b, d]) - (dist_matrix[a, b] + dist_matrix[c, d])
                
                if delta < -1e-10:
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    improved = True
                    
    return tour

def _or_opt(tour, dist_matrix, time_limit):
    """Improve the tour using Or-Opt moves (moving segments of size 1, 2, 3)."""
    n = len(tour)
    if n < 4:
        return tour
    
    tour = list(tour)
    t_start = time.time()
    improved = True
    
    while improved and (time.time() - t_start < time_limit):
        improved = False
        for seg_len in [1, 2, 3]:
            if time.time() - t_start > time_limit:
                return tour
            for i in range(n):
                if time.time() - t_start > time_limit:
                    return tour
                
                prev_i = (i - 1) % n
                next_i = (i + seg_len) % n
                
                segment = [tour[(i + k) % n] for k in range(seg_len)]
                first_node = segment[0]
                last_node = segment[-1]
                
                c_old1 = dist_matrix[tour[prev_i], first_node]
                c_old2 = dist_matrix[last_node, tour[next_i]]
                c_new_base = dist_matrix[tour[prev_i], tour[next_i]]
                
                for j in range(n):
                    if time.time() - t_start > time_limit:
                        return tour
                    
                    next_j = (j + 1) % n
                    
                    seg_set = set(segment)
                    if tour[j] in seg_set or tour[next_j] in seg_set:
                        continue
                    
                    c_old3 = dist_matrix[tour[j], tour[next_j]]
                    c_new2 = dist_matrix[tour[j], first_node]
                    c_new3 = dist_matrix[last_node, tour[next_j]]
                    
                    delta = (c_new_base + c_new2 + c_new3) - (c_old1 + c_old2 + c_old3)
                    
                    if delta < -1e-10:
                        new_tour = tour.copy()
                        del new_tour[i:i+seg_len]
                        
                        if j > i:
                            insert_idx = j - seg_len
                        else:
                            insert_idx = j
                        
                        new_tour[insert_idx:insert_idx] = segment
                        tour = new_tour
                        improved = True
                        
    return tour

def _compute_best_tour(dist_matrix):
    """
    Compute a high-quality tour using Multi-Start Greedy Insertion + 2-Opt + Or-Opt.
    """
    n = dist_matrix.shape[0]
    
    time_start = time.time()
    time_budget = _DEADLINE - time_start
    
    # Split time among multiple restarts
    # Let's do 3 restarts: Greedy Insertion (Start 0), Greedy Insertion (Start Random), NN (Start Random)
    
    best_tour = []
    best_length = float('inf')
    
    # Restart 1: Greedy Insertion from 0
    if time_budget > 10:
        tour = _greedy_insertion_tour(n, dist_matrix, 0)
        # Optimize
        t_rem = _DEADLINE - time.time()
        if t_rem > 5:
            tour = _two_opt(tour, dist_matrix, t_rem * 0.5)
        t_rem = _DEADLINE - time.time()
        if t_rem > 2:
            tour = _or_opt(tour, dist_matrix, t_rem * 0.5)
        
        length = sum(dist_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
        if length < best_length:
            best_length = length
            best_tour = tour
    
    # Restart 2: Nearest Neighbor from Random Start
    if time_budget > 5:
        import random
        random_start = random.randint(0, n - 1)
        tour = _nearest_neighbor_tour(n, dist_matrix, random_start)
        t_rem = _DEADLINE - time.time()
        if t_rem > 5:
            tour = _two_opt(tour, dist_matrix, t_rem * 0.5)
        t_rem = _DEADLINE - time.time()
        if t_rem > 2:
            tour = _or_opt(tour, dist_matrix, t_rem * 0.5)
        
        length = sum(dist_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
        if length < best_length:
            best_length = length
            best_tour = tour
            
    # Restart 3: If time remains, do another NN from a different random start
    if time.time() < _DEADLINE - 10:
        import random
        random_start = random.randint(0, n - 1)
        tour = _nearest_neighbor_tour(n, dist_matrix, random_start)
        t_rem = _DEADLINE - time.time()
        if t_rem > 5:
            tour = _two_opt(tour, dist_matrix, t_rem * 0.5)
        t_rem = _DEADLINE - time.time()
        if t_rem > 2:
            tour = _or_opt(tour, dist_matrix, t_rem * 0.5)
        
        length = sum(dist_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
        if length < best_length:
            best_length = length
            best_tour = tour

    return best_tour

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node by using a pre-computed optimized tour.
    """
    if len(unvisited_nodes) == 0:
        return -1
        
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Check time
    if time.time() > _DEADLINE:
        # Fallback to nearest neighbor
        dists = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists)
        return int(unvisited_nodes[best_idx])
    
    # Get or compute optimized tour
    fp = _get_dm_fingerprint(distance_matrix)
    
    if fp not in _tour_cache:
        # Compute optimized tour
        _tour_cache[fp] = _compute_best_tour(distance_matrix)
    
    optimized_tour = _tour_cache[fp]
    
    # Find the next node in the optimized tour after current_node
    try:
        idx = optimized_tour.index(current_node)
    except ValueError:
        # Fallback
        dists = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists)
        return int(unvisited_nodes[best_idx])
    
    next_node = optimized_tour[(idx + 1) % len(optimized_tour)]
    
    # Ensure the next node is in unvisited_nodes
    if next_node not in unvisited_nodes:
        unvisited_set = set(unvisited_nodes.tolist())
        for k in range(1, len(optimized_tour)):
            candidate = optimized_tour[(idx + k) % len(optimized_tour)]
            if candidate in unvisited_set:
                next_node = candidate
                break
        else:
            dists = distance_matrix[current_node, unvisited_nodes]
            best_idx = np.argmin(dists)
            return int(unvisited_nodes[best_idx])
    
    return int(next_node)
# EVOLVE-BLOCK-END
