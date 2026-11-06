import time
import random
import itertools

# EVOLVE-BLOCK-START

def _has_c4_bitmask(n, adj):
    """Check if graph has a C4 using bitmasks. Returns True if C4 found."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if common & (common - 1):  # at least two bits set
                return True
    return False

def _max_independent_set_size_bitmask(n, adj, target=11):
    """
    Check if there exists an independent set of size >= target.
    Returns the size of the maximum independent set found (capped at target).
    Uses branch and bound.
    """
    # Sort vertices by degree descending to improve pruning?
    # For bitmask, we just use the first set bit.
    
    def max_is(vertices_mask, remaining):
        if remaining <= 0:
            return 0
        if not vertices_mask:
            return 0
        
        # Upper bound: if number of vertices left < remaining, return count
        count = 0
        temp = vertices_mask
        while temp:
            temp &= temp - 1
            count += 1
        if count < remaining:
            return count
        
        # Pick a vertex to branch on - use first set bit
        lsb = vertices_mask & -vertices_mask
        v = lsb.bit_length() - 1
        
        # Option 1: include v
        new_mask = vertices_mask & ~(1 << v) & ~adj[v]
        include_val = 1 + max_is(new_mask, remaining - 1)
        
        # Option 2: exclude v
        exclude_val = max_is(vertices_mask ^ (1 << v), remaining)
        
        return max(include_val, exclude_val)
    
    all_vertices = (1 << n) - 1
    return max_is(all_vertices, target)

def _verify_graph(n, edges):
    """Verify the graph is C4-free and has no independent set of size 11."""
    if n < 1:
        return False
    adj = [0] * n
    for u, v in edges:
        if u == v:
            return False
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        if adj[u] & (1 << v):
            return False # Duplicate edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    if _has_c4_bitmask(n, adj):
        return False
    
    if _max_independent_set_size_bitmask(n, adj, 11) >= 11:
        return False
    
    return True

def _build_polarity_graph_q5():
    """
    Construct the polarity graph of PG(2,5).
    n = 31.
    """
    q = 5
    # Points of PG(2, q)
    points = []
    # Affine points: (1, x, y)
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    # Points at infinity: (0, 1, x)
    for x in range(q):
        points.append((0, 1, x))
    # Point at infinity: (0, 0, 1)
    points.append((0, 0, 1))
    
    n = len(points)
    adj = [0] * n
    edges = []
    
    # Polarity: u . v = 0 mod q
    # Using standard dot product
    for i in range(n):
        u = points[i]
        for j in range(i + 1, n):
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
            if dot == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append([i, j])
                
    return n, adj, edges

def _try_add_edge(adj, n, u, v):
    """Check if adding edge (u,v) creates a C4. Returns True if safe to add."""
    common = adj[u] & adj[v]
    return not (common & (common - 1))

def _construct_large_graph():
    """
    Try to construct a graph with n > 31.
    Starts with PG(2,5) polarity graph and tries to extend.
    """
    deadline = time.time() + 110
    
    # Base graph
    n_base, adj_base, edges_base = _build_polarity_graph_q5()
    
    # Check if base is valid (it should be)
    if not _verify_graph(n_base, edges_base):
        # Fallback to a known valid small graph if polarity fails (should not happen)
        # Disjoint triangles
        edges = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    best_result = {'n': n_base, 'edges': edges_base}
    
    # Try to extend to n = 39, 38, ..., 32
    # We will try random extensions from the base graph
    # To save time, we limit the number of restarts
    
    for target_n in range(39, n_base, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Try a few random extensions
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            random.seed(attempt * 1000 + target_n)
            
            adj = adj_base.copy()
            edges = edges_base.copy()
            n = n_base
            
            # Add (target_n - n_base) vertices
            # We need to assign neighbors for each new vertex
            # A new vertex v must connect to a subset S of existing vertices
            # such that no C4 is created.
            # Condition: For any u, w in existing, |N(u) cap N(w) cap S| <= 1.
            # Also, for any u in existing, N(u) cap S can be any size, 
            # BUT if u1, u2 in existing share a neighbor in S, that's fine.
            # The C4 risk is: u1 - s1 - u2 - s2 - u1.
            # So for any pair u1, u2, they cannot share 2 neighbors in S.
            
            success = True
            for _ in range(target_n - n_base):
                new_v = n
                n += 1
                adj.append(0)
                
                # Candidate neighbors: all existing vertices
                candidates = list(range(new_v))
                random.shuffle(candidates)
                
                chosen = []
                # Greedily pick neighbors for new_v
                # We must ensure that adding new_v with neighbor set S doesn't create C4.
                # This is equivalent to: for any u, w in existing, 
                # the number of common neighbors of u and w within S is <= 1.
                
                # Optimization: 
                # If we pick s1 in S, then for any u ~ s1, we cannot pick another s2 ~ u.
                # So N(u) \ {s1} becomes "forbidden" for future picks in S?
                # Yes. If u is adjacent to s1, then u cannot be adjacent to any other s2 in S.
                
                forbidden = [0] * new_v # Bitmask of forbidden vertices for next picks?
                # Actually, simpler:
                # Maintain a set of "blocked" vertices.
                # When we pick s, all neighbors of s in existing graph become blocked?
                # No. If we pick s, then for any u ~ s, u cannot have another neighbor in S.
                # So if u is adjacent to s, u is "saturated" for this new vertex's neighborhood?
                # No, u can be in S? No, S is subset of existing vertices.
                # u is an existing vertex.
                # If u ~ s, then u cannot be adjacent to any other s' in S.
                # So once we pick s, we must mark all neighbors of s as "cannot be picked again"?
                # No, we pick neighbors for new_v.
                # If new_v ~ s and new_v ~ s', and s ~ u and s' ~ u, then we have C4: new_v - s - u - s' - new_v.
                # So yes, if s and s' share a neighbor u in existing graph, they cannot both be in S.
                
                # So: We are selecting a set S of existing vertices such that no two vertices in S share a neighbor in the existing graph.
                # This is an "independent set" in the "neighborhood overlap" graph?
                # Or simply: S is a set of vertices in existing graph such that for any s1, s2 in S, N(s1) cap N(s2) = empty?
                # No, N(s1) cap N(s2) can be non-empty, but if u is in N(s1) cap N(s2), then u is a common neighbor.
                # The C4 is new_v - s1 - u - s2 - new_v.
                # So yes, we need N(s1) cap N(s2) to be empty for all s1, s2 in S.
                # Wait, is that true?
                # C4: new_v - s1 - u - s2 - new_v.
                # Edges: (new_v, s1), (s1, u), (u, s2), (s2, new_v).
                # s1, s2, u are existing vertices.
                # So if s1 and s2 share a neighbor u, we have a C4.
                # Therefore, S must be a set of vertices in the existing graph that are pairwise non-adjacent in the "line graph" of the existing graph?
                # No, just pairwise having no common neighbors.
                # This is a very strong constraint.
                
                # Let's re-evaluate.
                # If S has two vertices s1, s2 that share a neighbor u, C4 exists.
                # So S must be an independent set in the graph where edges connect vertices with common neighbors?
                # This set S is called a "2-independent" set?
                
                # Given the strong constraint, it might be hard to add many vertices.
                # Let's just try to pick a few.
                
                available = set(range(new_v))
                # We need to pick a subset S of available such that no two elements in S have a common neighbor in the existing graph.
                
                # Greedy:
                # Pick a random candidate s.
                # Remove all vertices that share a neighbor with s from available.
                
                chosen = []
                while available and len(chosen) < 5: # Limit degree of new vertex
                    s = random.choice(list(available))
                    chosen.append(s)
                    available.remove(s)
                    
                    # Find all vertices that share a neighbor with s
                    # N(s) in existing graph
                    ns = adj[s]
                    # For each u in N(s), all vertices in N(u) are forbidden
                    # So forbidden = union of N(u) for u in N(s)
                    # We also remove s itself (already done)
                    
                    # Calculate forbidden set
                    forb_mask = 0
                    temp = ns
                    while temp:
                        lsb = temp & -temp
                        u = lsb.bit_length() - 1
                        forb_mask |= adj[u]
                        temp ^= lsb
                    
                    # Remove forbidden from available
                    # available is a set of ints
                    new_available = set()
                    for v in available:
                        if not (forb_mask & (1 << v)):
                            new_available.add(v)
                    available = new_available
                    
                # Add edges
                for s in chosen:
                    adj[new_v] |= (1 << s)
                    adj[s] |= (1 << new_v)
                    edges.append([new_v, s])
                    
            # Check if this extension is valid
            if not _has_c4_bitmask(n, adj):
                if _max_independent_set_size_bitmask(n, adj, 11) < 11:
                    best_result = {'n': n, 'edges': edges.copy()}
                    found = True
                    break
                    
        if found:
            # If we found a valid graph for target_n, we can stop or try higher?
            # We are iterating down from 39. If we find 39, we stop.
            # If we find 38, we might try 39? No, we started at 39.
            # So if we found one for the current target_n, we return it?
            # Actually, we should keep the best one found so far.
            # Since we go 39, 38, ..., the first one found is the largest.
            break
            
    return best_result

def construct():
    """Construct a C4-free graph with no independent set of size 11, maximizing n."""
    start_time = time.time()
    
    # Try to construct a large graph
    result = _construct_large_graph()
    
    # Verify the result
    if _verify_graph(result['n'], result['edges']):
        return result
    
    # Fallback: polarity graph of PG(2,5) (31 vertices, C4-free)
    n_base, adj_base, edges_base = _build_polarity_graph_q5()
    return {'n': n_base, 'edges': edges_base}

# EVOLVE-BLOCK-END
