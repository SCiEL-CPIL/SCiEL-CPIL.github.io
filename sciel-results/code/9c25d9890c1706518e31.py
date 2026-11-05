import time
import random
import sys

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

def _has_independent_set(adj, n, limit=11):
    """
    Check if there is an independent set of size >= limit.
    Uses branch and bound with bitmasks for speed.
    """
    if n <= 64:
        adj_mask = [0] * n
        for i in range(n):
            mask = 0
            for j in adj[i]:
                mask |= (1 << j)
            adj_mask[i] = mask
        
        all_mask = (1 << n) - 1
        
        def _branch(rem_mask, current_size):
            if current_size >= limit:
                return True
            if rem_mask == 0:
                return False
            if current_size + rem_mask.bit_count() < limit:
                return False
            
            # Choose vertex with maximum degree in remaining graph
            best_v = -1
            best_deg = -1
            m = rem_mask
            while m:
                v = (m & -m).bit_length() - 1
                m &= m - 1
                deg = (adj_mask[v] & rem_mask).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
            
            # Try including best_v
            rem_include = rem_mask & ~(adj_mask[best_v] | (1 << best_v))
            if _branch(rem_include, current_size + 1):
                return True
            
            # Try excluding best_v
            rem_exclude = rem_mask & ~(1 << best_v)
            if _branch(rem_exclude, current_size):
                return True
            
            return False
        
        return _branch(all_mask, 0)
    else:
        remaining = set(range(n))
        
        def _branch(rem, current_size):
            if current_size >= limit:
                return True
            if not rem:
                return False
            if current_size + len(rem) < limit:
                return False
            
            best_v = -1
            best_deg = -1
            for v in rem:
                deg = len(adj[v] & rem)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
                
            rem_include = rem - adj[best_v] - {best_v}
            if _branch(rem_include, current_size + 1):
                return True
            
            rem_exclude = rem - {best_v}
            if _branch(rem_exclude, current_size):
                return True
            
            return False
        
        return _branch(remaining, 0)

def _build_conflict_masks(adj, n):
    """
    Build conflict masks: conflict_mask[i] has bit j set if N(i) ∩ N(j) ≠ ∅.
    A valid neighborhood for a new vertex is an independent set in the conflict graph.
    """
    conflict_mask = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & adj[j]:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    return conflict_mask

def _find_max_independent_set_in_conflict(conflict_mask, n, limit=None):
    """
    Find a large independent set in the conflict graph.
    If limit is given, stop when we find one of size >= limit.
    Returns the set of vertices in the independent set.
    """
    all_mask = (1 << n) - 1
    
    # Greedy approach with multiple random orders
    best_set = set()
    
    for attempt in range(50):
        # Random order
        order = list(range(n))
        random.shuffle(order)
        
        current_set = set()
        for v in order:
            # Check if v is compatible with all in current_set
            # v is compatible with u if conflict_mask[v] has no bit set for u
            compatible = True
            for u in current_set:
                if conflict_mask[v] & (1 << u):
                    compatible = False
                    break
            if compatible:
                current_set.add(v)
        
        if len(current_set) > len(best_set):
            best_set = current_set
            if limit and len(best_set) >= limit:
                break
    
    return best_set

def construct():
    deadline = time.time() + 145.0
    best_n = 0
    best_edges = []
    
    # Start with q=5 polarity graph (31 vertices)
    n, edges, adj = _construct_polarity_graph(5)
    
    # Verify
    if not _verify_c4_free(n, adj):
        return {'n': 0, 'edges': []}
        
    if _has_independent_set(adj, n, limit=11):
        return {'n': 0, 'edges': []}
        
    best_n = n
    best_edges = edges[:]
    
    # Try to add vertices one by one
    max_additional = 12  # Target up to 43
    
    for add_iter in range(max_additional):
        if time.time() > deadline - 15:
            break
            
        added = False
        
        # Build conflict masks for the current graph
        conflict_mask = _build_conflict_masks(adj, n)
        
        # Try to find a good neighborhood S for the new vertex
        # We want S to be large (to keep alpha low) but valid (independent in conflict graph)
        
        # Try multiple strategies
        for strategy in range(20):
            if time.time() > deadline - 10:
                break
                
            # Strategy: find a large independent set in the conflict graph
            S = _find_max_independent_set_in_conflict(conflict_mask, n, limit=None)
            
            # If S is too small, the new vertex will have low degree, which might increase alpha
            # We need S to be large enough that the new vertex doesn't create a large independent set
            
            # Add new vertex with neighborhood S
            new_adj = [set(s) for s in adj]
            new_adj.append(set(S))
            for u in S:
                new_adj[u].add(n)
                
            # Check alpha of new graph
            if not _has_independent_set(new_adj, n + 1, limit=11):
                # Valid! Add this vertex
                adj = new_adj
                n += 1
                for u in S:
                    edges.append([n - 1, u])
                
                if n > best_n:
                    best_n = n
                    best_edges = [e[:] for e in edges]
                    
                added = True
                break
        
        if not added:
            # Try different random seeds
            for retry in range(10):
                if time.time() > deadline - 10:
                    break
                    
                # Shuffle and try again
                conflict_mask = _build_conflict_masks(adj, n)
                S = _find_max_independent_set_in_conflict(conflict_mask, n, limit=None)
                
                new_adj = [set(s) for s in adj]
                new_adj.append(set(S))
                for u in S:
                    new_adj[u].add(n)
                    
                if not _has_independent_set(new_adj, n + 1, limit=11):
                    adj = new_adj
                    n += 1
                    for u in S:
                        edges.append([n - 1, u])
                    
                    if n > best_n:
                        best_n = n
                        best_edges = [e[:] for e in edges]
                        
                    added = True
                    break
            
            if not added:
                break
    
    return {'n': best_n, 'edges': best_edges}
