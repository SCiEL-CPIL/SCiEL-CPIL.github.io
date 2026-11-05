import time
import random

def _construct_polarity_graph(q):
    """
    Construct the polarity graph of the projective plane PG(2, q).
    Vertices are points of the plane.
    Two points p and r are adjacent if p lies on the polar line of r.
    This graph is C4-free.
    """
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, q-2, q)
                    key = (1, (y * inv_x) % q, (z * inv_x) % q)
                elif y != 0:
                    inv_y = pow(y, q-2, q)
                    key = (0, 1, (z * inv_y) % q)
                else:
                    key = (0, 0, 1)
                
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    n_pts = len(points)
    
    adj = [set() for _ in range(n_pts)]
    
    for i in range(n_pts):
        x1, x2, x3 = points[i]
        for j in range(i + 1, n_pts):
            y1, y2, y3 = points[j]
            if (x1 * y1 + x2 * y2 + x3 * y3) % q == 0:
                adj[i].add(j)
                adj[j].add(i)
                
    edges = []
    for i in range(n_pts):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
                
    return n_pts, edges, adj

def _verify_c4_free(n, adj):
    """Check that no two vertices have >= 2 common neighbours."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if len(common) >= 2:
                return False
    return True

def _has_independent_set(adj_mask, n, limit=11):
    """
    Check if there is an independent set of size >= limit.
    Uses bitmasks for speed with branch and bound.
    """
    all_mask = (1 << n) - 1
    
    def _branch(rem_mask, current_size):
        if current_size >= limit:
            return True
        if rem_mask == 0:
            return False
        if current_size + rem_mask.bit_count() < limit:
            return False
            
        # Pick vertex with max degree in remaining set
        best_v = -1
        best_deg = -1
        m = rem_mask
        while m:
            v_bit = m & -m
            v = v_bit.bit_length() - 1
            m ^= v_bit
            deg = (adj_mask[v] & rem_mask).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
                
        if best_v == -1:
            return False
            
        # Branch 1: Include best_v (remove it and its neighbors)
        rem_include = rem_mask & ~(adj_mask[best_v] | (1 << best_v))
        if _branch(rem_include, current_size + 1):
            return True
            
        # Branch 2: Exclude best_v
        rem_exclude = rem_mask & ~(1 << best_v)
        if _branch(rem_exclude, current_size):
            return True
            
        return False

    return _branch(all_mask, 0)

def _find_valid_neighbor_set(adj, n):
    """
    Find a set of neighbors S for a new vertex such that
    for all u, v in S, N(u) ∩ N(v) = ∅ (to maintain C4-freeness).
    Uses greedy approach with random ordering.
    """
    candidates = list(range(n))
    random.shuffle(candidates)
    
    S = set()
    for v in candidates:
        valid = True
        for u in S:
            if adj[u] & adj[v]:
                valid = False
                break
        if valid:
            S.add(v)
            
    return S

def construct():
    deadline = time.time() + 135.0  # Safety margin before 150s kill
    best_n = 0
    best_edges = []
    
    # Start with q=5 polarity graph (31 vertices)
    n, edges, adj = _construct_polarity_graph(5)
    
    # Verify C4-free
    if not _verify_c4_free(n, adj):
        # Fallback: try q=7
        n, edges, adj = _construct_polarity_graph(7)
        if not _verify_c4_free(n, adj):
            return {'n': 3, 'edges': [[0, 1], [1, 2], [0, 2]]}
    
    # Check alpha of base graph
    adj_mask = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        adj_mask[i] = mask
    
    if _has_independent_set(adj_mask, n, limit=11):
        # Base graph already has alpha >= 11, need to find subset or use different approach
        # For now, try to find a valid subset
        for target_n in range(43, max(n-1, 30), -1):
            if time.time() > deadline - 10:
                break
            for attempt in range(1000):
                if time.time() > deadline - 5:
                    break
                selected = random.sample(range(n), target_n)
                selected_set = set(selected)
                old_to_new = {old: new for new, old in enumerate(selected)}
                
                sub_adj_mask = [0] * target_n
                for old_i in selected:
                    new_i = old_to_new[old_i]
                    for old_j in adj[old_i]:
                        if old_j in selected_set:
                            new_j = old_to_new[old_j]
                            sub_adj_mask[new_i] |= (1 << new_j)
                
                if not _has_independent_set(sub_adj_mask, target_n, limit=11):
                    candidate_edges = []
                    for i in range(target_n):
                        mask = sub_adj_mask[i]
                        while mask:
                            bit = mask & -mask
                            j = bit.bit_length() - 1
                            if i < j:
                                candidate_edges.append([i, j])
                            mask ^= bit
                    best_n = target_n
                    best_edges = candidate_edges
                    break
            if best_n > 0:
                break
        if best_n > 0:
            return {'n': best_n, 'edges': best_edges}
        return {'n': 3, 'edges': [[0, 1], [1, 2], [0, 2]]}
    
    # Base graph is valid (alpha < 11), now try to grow it
    best_n = n
    best_edges = [e[:] for e in edges]
    
    # Try to add vertices one by one
    for step in range(12):  # Try to add up to 12 more vertices
        if time.time() > deadline - 15:
            break
            
        added = False
        # Try multiple random attempts for neighbor selection
        for attempt in range(100):
            if time.time() > deadline - 10:
                break
                
            # Generate a candidate neighbor set
            S = _find_valid_neighbor_set(adj, n)
            
            # Add new vertex with neighbors S
            new_adj = [set(s) for s in adj]
            new_adj.append(set(S))
            for u in S:
                new_adj[u].add(n)
                
            # Check C4-free (should be guaranteed by construction, but verify)
            if not _verify_c4_free(n + 1, new_adj):
                continue
                
            # Build adj_mask for new graph
            new_adj_mask = [0] * (n + 1)
            for i in range(n + 1):
                mask = 0
                for j in new_adj[i]:
                    mask |= (1 << j)
                new_adj_mask[i] = mask
            
            # Check alpha
            if not _has_independent_set(new_adj_mask, n + 1, limit=11):
                # Valid! Update state
                adj = new_adj
                n += 1
                for u in S:
                    edges.append([n - 1, u])
                
                if n > best_n:
                    best_n = n
                    best_edges = [e[:] for e in edges]
                    
                added = True
                break  # Restart with new graph
        
        if not added:
            break  # Can't add more vertices
    
    # If we didn't reach a good size, try q=7 subsets as fallback
    if best_n < 39:
        n7, edges7, adj7 = _construct_polarity_graph(7)
        if _verify_c4_free(n7, adj7):
            adj_mask7 = [0] * n7
            for i in range(n7):
                mask = 0
                for j in adj7[i]:
                    mask |= (1 << j)
                adj_mask7[i] = mask
            
            # Try to find a valid subset of size 39-43
            for target_n in [43, 42, 41, 40, 39]:
                if target_n <= best_n or target_n > n7:
                    continue
                if time.time() > deadline - 10:
                    break
                    
                found = False
                for attempt in range(2000):
                    if time.time() > deadline - 5:
                        break
                    selected = random.sample(range(n7), target_n)
                    selected_set = set(selected)
                    old_to_new = {old: new for new, old in enumerate(selected)}
                    
                    sub_adj_mask = [0] * target_n
                    for old_i in selected:
                        new_i = old_to_new[old_i]
                        for old_j in adj7[old_i]:
                            if old_j in selected_set:
                                new_j = old_to_new[old_j]
                                sub_adj_mask[new_i] |= (1 << new_j)
                    
                    if not _has_independent_set(sub_adj_mask, target_n, limit=11):
                        candidate_edges = []
                        for i in range(target_n):
                            mask = sub_adj_mask[i]
                            while mask:
                                bit = mask & -mask
                                j = bit.bit_length() - 1
                                if i < j:
                                    candidate_edges.append([i, j])
                                mask ^= bit
                        best_n = target_n
                        best_edges = candidate_edges
                        found = True
                        break
                if found:
                    break
    
    if best_n == 0:
        best_n = 3
        best_edges = [[0, 1], [1, 2], [0, 2]]
        
    return {'n': best_n, 'edges': best_edges}
