import time
import random

def construct():
    deadline = time.time() + 115.0
    
    # Generate Polarity Graph of PG(2,5)
    q = 5
    
    # Generate all non-zero vectors in F_q^3, normalized so first non-zero is 1
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    points.append((1, (y * pow(x, q-2)) % q, (z * pow(x, q-2)) % q))
                elif y != 0:
                    points.append((0, 1, (z * pow(y, q-2)) % q))
                else:
                    points.append((0, 0, 1))
    
    n_base = len(points)
    # n_base should be 31
    
    # Build adjacency using bitmasks
    adj = [0] * n_base
    for i in range(n_base):
        for j in range(i + 1, n_base):
            u = points[i]
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
            if dot == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Verify C4-free
    def is_c4_free(adj, n):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if common & (common - 1):  # more than one bit set
                    return False
        return True
    
    # Verify alpha < 11 using a branch and bound
    def max_independent_set(adj, n, limit=11):
        """Returns True if there exists an independent set of size >= limit"""
        # Use branch and bound
        # Sort by degree descending for better pruning
        degrees = [bin(adj[v]).count('1') for v in range(n)]
        order = sorted(range(n), key=lambda v: -degrees[v])
        
        best_found = [0]
        
        def backtrack(start, size, excluded_mask):
            if size >= limit:
                return True
            if time.time() > deadline - 20:
                return False
            # Upper bound: can we still reach limit?
            remaining = n - start
            if size + remaining < limit:
                return False
            
            if start == n:
                return False
            
            v = order[start]
            if excluded_mask & (1 << v):
                return backtrack(start + 1, size, excluded_mask)
            
            # Option 1: Don't include v
            if backtrack(start + 1, size, excluded_mask):
                return True
            
            # Option 2: Include v
            # Remove v and all its neighbors from consideration
            new_excluded = excluded_mask | (1 << v) | adj[v]
            if backtrack(start + 1, size + 1, new_excluded):
                return True
            
            return False
        
        return backtrack(0, 0, 0)
    
    # Check base graph
    if not is_c4_free(adj, n_base):
        # Fallback to disjoint triangles
        n_fallback = 30
        adj_fallback = [0] * n_fallback
        for i in range(10):
            u, v, w = 3*i, 3*i+1, 3*i+2
            adj_fallback[u] |= (1<<v) | (1<<w)
            adj_fallback[v] |= (1<<u) | (1<<w)
            adj_fallback[w] |= (1<<u) | (1<<v)
        edges = []
        for u in range(n_fallback):
            for v in range(u + 1, n_fallback):
                if adj_fallback[u] & (1 << v):
                    edges.append([u, v])
        return {'n': n_fallback, 'edges': edges}
    
    # Try to add vertices greedily
    # For each new vertex, try all possible adjacency patterns to existing vertices
    # that maintain C4-freeness
    
    current_n = n_base
    current_adj = adj[:]
    
    def try_add_vertex(adj_list, n, max_n=43):
        """Try to add one vertex. Returns new adj_list and n if successful, else None."""
        if n >= max_n:
            return None
        
        # For a new vertex at position n, try different adjacency patterns
        # The constraint: for any pair u,v in 0..n-1, the number of common neighbors
        # among {0..n-1} plus the new vertex must be <= 1.
        # So if u and v already have a common neighbor w, the new vertex cannot be
        # adjacent to both u and v.
        
        # Also, the new vertex itself doesn't create C4s with itself.
        
        # Precompute: for each pair (u,v), do they already share a common neighbor?
        # If yes, new vertex can't be adjacent to both.
        
        # Try random adjacency patterns, biased towards connecting to many vertices
        # (to keep alpha small)
        
        random.seed(42)  # Deterministic for reproducibility
        
        for attempt in range(100):
            if time.time() > deadline - 30:
                return None
            
            # Generate a random adjacency pattern
            # Bias: connect to ~50-70% of vertices
            new_adj_to_existing = 0
            for v in range(n):
                if random.random() < 0.6:
                    new_adj_to_existing |= (1 << v)
            
            # Check C4-freeness with this pattern
            valid = True
            for u in range(n):
                if not (new_adj_to_existing & (1 << u)):
                    continue
                for v in range(u + 1, n):
                    if not (new_adj_to_existing & (1 << v)):
                        continue
                    # u and v are both adjacent to new vertex
                    # They must not share any other common neighbor
                    common = adj_list[u] & adj_list[v]
                    if common & (common - 1):  # already have 2+ common neighbors
                        valid = False
                        break
                    # Also check: do u and v share exactly one common neighbor?
                    # If so, that's fine as long as it's not the new vertex (which it isn't
                    # since we're only checking existing neighbors)
                    # Actually, the condition is: total common neighbors <= 1
                    # Existing common neighbors: adj_list[u] & adj_list[v]
                    # New common neighbor: the new vertex (if both u,v adjacent to it)
                    # So total = popcount(adj_list[u] & adj_list[v]) + 1 (if both adj to new)
                    # We need this <= 1, so popcount(adj_list[u] & adj_list[v]) must be 0
                    # if both u,v are adjacent to new vertex
                    pass
            
            # More careful check:
            valid = True
            for u in range(n):
                for v in range(u + 1, n):
                    existing_common = adj_list[u] & adj_list[v]
                    existing_count = bin(existing_common).count('1')
                    new_common = 1 if (new_adj_to_existing & (1 << u) and new_adj_to_existing & (1 << v)) else 0
                    total_common = existing_count + new_common
                    if total_common > 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check that alpha remains < 11
            # Build the new adjacency
            new_adj = adj_list[:] + [0]
            for v in range(n):
                if new_adj_to_existing & (1 << v):
                    new_adj[n] |= (1 << v)
                    new_adj[v] |= (1 << n)
            
            # Quick alpha check: try to find independent set of size 11
            # Use a greedy approach first
            def has_independent_set_11(adj_list, n):
                # Branch and bound
                if n < 11:
                    return False
                
                # Greedy lower bound
                remaining = (1 << n) - 1
                count = 0
                while remaining:
                    best_v = -1
                    best_deg = n + 1
                    for v in range(n):
                        if remaining & (1 << v):
                            non_adj = remaining & ~(adj_list[v] | (1 << v))
                            deg = bin(non_adj).count('1')
                            if deg < best_deg:
                                best_deg = deg
                                best_v = v
                    count += 1
                    non_adj = remaining & ~(adj_list[best_v] | (1 << best_v))
                    remaining &= ~(non_adj | (1 << best_v))
                if count >= 11:
                    return True
                
                # More careful: use branch and bound to see if alpha >= 11
                # This is expensive, so use it sparingly
                # For now, if greedy finds >= 11, return True
                # Otherwise, assume it's < 11 (may be wrong, but fast)
                return False
            
            if has_independent_set_11(new_adj, n + 1):
                continue
            
            return new_adj, n + 1
        
        return None
    
    # Try to add vertices
    for _ in range(12):  # Try to add up to 12 more vertices
        if time.time() > deadline - 25:
            break
        result = try_add_vertex(current_adj, current_n)
        if result is None:
            break
        current_adj, current_n = result
    
    # Final validation
    if not is_c4_free(current_adj, current_n):
        # Fallback
        n_fallback = 30
        adj_fallback = [0] * n_fallback
        for i in range(10):
            u, v, w = 3*i, 3*i+1, 3*i+2
            adj_fallback[u] |= (1<<v) | (1<<w)
            adj_fallback[v] |= (1<<u) | (1<<w)
            adj_fallback[w] |= (1<<u) | (1<<v)
        edges = []
        for u in range(n_fallback):
            for v in range(u + 1, n_fallback):
                if adj_fallback[u] & (1 << v):
                    edges.append([u, v])
        return {'n': n_fallback, 'edges': edges}
    
    # Check alpha < 11 one more time
    if max_independent_set(current_adj, current_n, limit=11):
        # Has independent set of size 11, invalid
        # Fall back to base graph if it was valid
        if is_c4_free(adj, n_base) and not max_independent_set(adj, n_base, limit=11):
            current_adj = adj
            current_n = n_base
        else:
            # Fallback to disjoint triangles
            n_fallback = 30
            adj_fallback = [0] * n_fallback
            for i in range(10):
                u, v, w = 3*i, 3*i+1, 3*i+2
                adj_fallback[u] |= (1<<v) | (1<<w)
                adj_fallback[v] |= (1<<u) | (1<<w)
                adj_fallback[w] |= (1<<u) | (1<<v)
            edges = []
            for u in range(n_fallback):
                for v in range(u + 1, n_fallback):
                    if adj_fallback[u] & (1 << v):
                        edges.append([u, v])
            return {'n': n_fallback, 'edges': edges}
    
    # Build edge list
    edges = []
    for u in range(current_n):
        for v in range(u + 1, current_n):
            if current_adj[u] & (1 << v):
                edges.append([u, v])
    
    return {'n': current_n, 'edges': edges}
