import time
import random
import sys

# EVOLVE-BLOCK-START
_DEADLINE = 118.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _deadline_hit():
    return _elapsed() >= _DEADLINE

def _build_polarity_graph(q):
    """Construct the polarity graph of the projective plane PG(2, q)."""
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
                
    return adj

def _has_indep_11(adj):
    """Check if there is an independent set of size 11."""
    n = len(adj)
    if n < 11:
        return False
        
    masks = []
    for i in range(n):
        m = 0
        for j in adj[i]:
            m |= (1 << j)
        masks.append(m)
    
    sys.setrecursionlimit(10000)
    
    def branch(remaining_mask, current_size):
        if current_size >= 11:
            return True
        if remaining_mask == 0:
            return False
        # Upper bound
        if current_size + remaining_mask.bit_count() < 11:
            return False
        
        # Pick vertex with highest degree in remaining
        rem = remaining_mask
        best_v = -1
        best_deg = -1
        while rem:
            lsb = rem & (-rem)
            v = lsb.bit_length() - 1
            rem ^= lsb
            deg = (remaining_mask & masks[v]).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
        
        if best_v == -1:
            return False

        # Branch include
        new_rem = remaining_mask & ~(masks[best_v] | (1 << best_v))
        if branch(new_rem, current_size + 1):
            return True
        
        # Branch exclude
        new_rem2 = remaining_mask & ~(1 << best_v)
        if branch(new_rem2, current_size):
            return True
        
        return False
    
    full = (1 << n) - 1
    return branch(full, 0)

def _get_conflict_masks(adj, n):
    """
    Returns a list of bitmasks for the conflict graph.
    conflict_mask[i] has bit j set if i and j share a common neighbor in adj.
    """
    # First, compute common neighbor masks for all pairs?
    # Or just compute conflict masks directly.
    # conflict_mask[i] = union of (N(j) & N(i)) for j in N(i)? No.
    # u and v conflict if N(u) & N(v) is not empty.
    
    # Efficient way:
    # For each vertex k, let N(k) be its neighbors.
    # For any pair u, v in N(k), u and v have k as a common neighbor.
    # So u and v conflict.
    # Thus, for each k, all pairs in N(k) are edges in the conflict graph.
    
    conflict_masks = [0] * n
    for k in range(n):
        neighbors = list(adj[k])
        # For each pair u, v in neighbors, set conflict
        # This is O(sum deg^2).
        # For q=5, deg is ~6. 31 * 36 = 1116 ops. Very fast.
        for i in range(len(neighbors)):
            u = neighbors[i]
            for j in range(i+1, len(neighbors)):
                v = neighbors[j]
                conflict_masks[u] |= (1 << v)
                conflict_masks[v] |= (1 << u)
                
    return conflict_masks

def _find_independent_set_in_conflict(conflict_masks, n, target_size=None):
    """
    Find an independent set in the graph defined by conflict_masks.
    Independent set in conflict graph = set of vertices no two of which share a common neighbor.
    We use a greedy approach with randomization.
    """
    remaining = (1 << n) - 1
    independent_set = []
    
    # Greedy: pick vertex with min degree in remaining conflict graph
    while remaining:
        # Calculate degree of each vertex in remaining
        best_v = -1
        best_deg = 10**9
        
        m = remaining
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            m ^= lsb
            # Degree of v in remaining conflict graph
            deg = (conflict_masks[v] & remaining).bit_count()
            if deg < best_deg:
                best_deg = deg
                best_v = v
        
        if best_v == -1:
            break
            
        independent_set.append(best_v)
        # Remove best_v and its neighbors from remaining
        # In conflict graph, neighbors of v are those that conflict with v.
        # We need to remove v and all u such that u conflicts with v.
        # conflict_masks[v] gives the mask of vertices conflicting with v.
        remaining &= ~(conflict_masks[v] | (1 << best_v))
        
    return independent_set

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = 118.0
    
    # Start with PG(2,5)
    adj = _build_polarity_graph(5)
    n = len(adj)
    
    # Verify initial graph
    if _has_indep_11(adj):
        # Fallback to a known valid smaller graph if PG(2,5) fails (unlikely)
        # Construct a small C4-free graph manually
        # 30 vertices: 10 triangles? No, triangles have C4? No.
        # But triangles have common neighbors?
        # In a triangle, any two vertices have 1 common neighbor.
        # So a disjoint union of triangles is C4-free.
        # 10 triangles -> 30 vertices.
        # Alpha = 10 (one from each triangle).
        # We need alpha < 11. So 10 triangles is valid.
        edges = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
        
    best_n = n
    best_adj = [set(s) for s in adj]
    
    # Main expansion loop
    while _elapsed() < 100.0: # Leave 18s for final check and output
        # Compute conflict masks
        conflict_masks = _get_conflict_masks(adj, n)
        
        added = False
        # Try to find a valid independent set in conflict graph
        # We want a large set to give the new vertex high degree
        # Try multiple randomizations or just greedy
        
        # To add diversity, we can try different vertices as "seeds" for the greedy
        # Or just run greedy once. If it fails alpha check, try again with different randomization?
        # Greedy is deterministic.
        # Let's try a few random restarts of the greedy by shuffling the order of vertex consideration?
        # Actually, the greedy picks min degree.
        # We can try to pick a random vertex with low degree.
        
        found_valid = False
        for attempt in range(50):
            if _deadline_hit():
                break
                
            # Generate a candidate independent set S in conflict graph
            # Modify greedy to be slightly random
            remaining = (1 << n) - 1
            S = []
            
            while remaining:
                # Pick vertex with min degree, tie-break randomly
                candidates = []
                min_deg = 10**9
                m = remaining
                while m:
                    lsb = m & (-m)
                    v = lsb.bit_length() - 1
                    m ^= lsb
                    deg = (conflict_masks[v] & remaining).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        candidates = [v]
                    elif deg == min_deg:
                        candidates.append(v)
                        
                if not candidates:
                    break
                    
                v = random.choice(candidates)
                S.append(v)
                remaining &= ~(conflict_masks[v] | (1 << v))
                
            if not S:
                continue
                
            # Try adding a new vertex connected to S
            # Check if this addition maintains C4-free
            # By construction of S (independent in conflict graph), no two vertices in S share a common neighbor.
            # So C4-free is maintained.
            
            # Check Alpha
            new_n = n + 1
            new_adj = [set(s) for s in adj]
            new_adj.append(set(S))
            for u in S:
                new_adj[u].add(new_n - 1)
                
            if not _has_indep_11(new_adj):
                # Valid
                adj = new_adj
                n = new_n
                if n > best_n:
                    best_n = n
                    best_adj = [set(s) for s in adj]
                added = True
                found_valid = True
                break # Restart loop with new graph
            
        if not found_valid:
            # If we couldn't add a vertex in this iteration, maybe try removing some edges?
            # Or just break if we've been stuck for a while?
            # For now, break to avoid wasting time.
            # We could try a different strategy, but let's keep it simple.
            break
            
    # Final verification
    if _check_c4(best_adj) and not _has_indep_11(best_adj):
        n = len(best_adj)
        edges = []
        for i in range(n):
            for j in best_adj[i]:
                if i < j:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
    else:
        # Fallback
        edges = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}

def _check_c4(adj):
    """adj: list of sets. Return True if no C4."""
    n = len(adj)
    for i in range(n):
        for j in range(i+1, n):
            if len(adj[i] & adj[j]) >= 2:
                return False
    return True

# EVOLVE-BLOCK-END
