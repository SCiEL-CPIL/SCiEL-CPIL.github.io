"""Improved: fast K4/ind-6 detection + seeded local search from known 35-vertex graph."""
import time
import random

_DEADLINE = None

def _k4_found(adj, n):
    """Return a 4-tuple (a,b,c,d) forming a K4, or None. adj = list of int bitmasks."""
    for a in range(n):
        aa = adj[a]
        for b in range(a + 1, n):
            if not (aa >> b) & 1:
                continue
            ab = aa & adj[b]
            for c in range(b + 1, n):
                if not (ab >> c) & 1:
                    continue
                abc = ab & adj[c]
                for d in range(c + 1, n):
                    if (abc >> d) & 1:
                        return (a, b, c, d)
    return None

def _ind6_found(adj, n):
    """Return a 6-tuple forming an independent set, or None. adj = list of int bitmasks."""
    allmask = (1 << n) - 1
    for a in range(n):
        na = allmask ^ (1 << a) ^ adj[a]
        for b in range(a + 1, n):
            if not (na >> b) & 1:
                continue
            nab = na & (allmask ^ (1 << b) ^ adj[b])
            for c in range(b + 1, n):
                if not (nab >> c) & 1:
                    continue
                nabc = nab & (allmask ^ (1 << c) ^ adj[c])
                for d in range(c + 1, n):
                    if not (nabc >> d) & 1:
                        continue
                    nabcd = nabc & (allmask ^ (1 << d) ^ adj[d])
                    # For each pair (e, f) with e < f, both must be in nabcd and not adjacent
                    # nabcd has bits only for vertices > d
                    # We need two non-adjacent vertices from nabcd
                    # Iterate e from d+1 to n-1
                    for e in range(d + 1, n):
                        if not (nabcd >> e) & 1:
                            continue
                        # rest = vertices > e that are non-adjacent to e and in nabcd
                        rest = nabcd & (allmask ^ (1 << e) ^ adj[e])
                        # Need at least one vertex > e in rest
                        higher = rest & (~((1 << (e + 1)) - 1))
                        if higher:
                            f = (higher & -higher).bit_length() - 1
                            return (a, b, c, d, e, f)
    return None

def _is_valid(adj, n):
    return _k4_found(adj, n) is None and _ind6_found(adj, n) is None

def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return edges

def _build_35_vertex_graph():
    """Construct the known 35-vertex (4,6)-Ramsey graph from literature."""
    n = 35
    adj = [0] * n
    # Use a structured construction based on Z_7 x Z_5
    # Vertex (i, j) where i in 0..6, j in 0..4
    # Edge between (i1,j1) and (i2,j2) if (i2-i1 mod 7) in {1,2,4} AND (j2-j1 mod 5) in {1,2,3,4}
    # This is a Cayley graph construction
    S7 = {1, 2, 4}  # quadratic residues mod 7
    S5 = {1, 2, 3, 4}  # all nonzero mod 5
    
    for i1 in range(7):
        for j1 in range(5):
            v1 = i1 * 5 + j1
            for i2 in range(i1 + 1, 7):
                for j2 in range(5):
                    v2 = i2 * 5 + j2
                    di = (i2 - i1) % 7
                    dj = (j2 - j1) % 5
                    if di in S7 and dj in S5:
                        adj[v1] |= (1 << v2)
                        adj[v2] |= (1 << v1)
            # Also check same i, different j
            for j2 in range(j1 + 1, 5):
                v2 = i1 * 5 + j2
                dj = (j2 - j1) % 5
                di = 0
                # No edge if di=0 since 0 not in S7
    return adj

def _try_extend_35_to_36(adj35, deadline):
    """Try to extend the 35-vertex graph to 36 vertices by adding one vertex."""
    n_old = 35
    n_new = 36
    
    # Strategy 1: Constraint-based enumeration of the new vertex's neighborhood
    # For the new vertex v=35, its neighborhood S must satisfy:
    #   - No K4: for every triangle (a,b,c) in G35, not all of a,b,c in S
    #   - No independent 6-set: for every independent 5-set (a,b,c,d,e) in G35,
    #     at least one of a,b,c,d,e is in S (so that v + those 5 doesn't form ind-6
    #     unless v is non-adjacent to all 5... wait, if v is adjacent to at least one
    #     of the 5, then v+5 is not independent)
    # Actually: v + {a,b,c,d,e} is independent iff v is non-adjacent to all of a,b,c,d,e
    # AND {a,b,c,d,e} is independent in G35.
    # So the constraint is: for every independent 5-set I in G35, S must intersect I
    # (i.e., v must be adjacent to at least one vertex in I).
    #
    # Also: v + {a,b,c} forms a K4 iff {a,b,c} is a triangle in G35 AND v is adjacent
    # to all of a,b,c. So: for every triangle T in G35, S must NOT contain all of T.
    
    # Precompute all triangles and all independent 5-sets in G35
    allmask = (1 << n_old) - 1
    
    # Find all triangles
    triangles = []
    for a in range(n_old):
        aa = adj35[a]
        for b in range(a + 1, n_old):
            if not (aa >> b) & 1:
                continue
            ab = aa & adj35[b]
            for c in range(b + 1, n_old):
                if (ab >> c) & 1:
                    triangles.append((a, b, c))
    
    # Find all independent 5-sets
    ind5sets = []
    for a in range(n_old):
        na = allmask ^ (1 << a) ^ adj35[a]
        for b in range(a + 1, n_old):
            if not (na >> b) & 1:
                continue
            nab = na & (allmask ^ (1 << b) ^ adj35[b])
            for c in range(b + 1, n_old):
                if not (nab >> c) & 1:
                    continue
                nabc = nab & (allmask ^ (1 << c) ^ adj35[c])
                for d in range(c + 1, n_old):
                    if not (nabc >> d) & 1:
                        continue
                    nabcd = nabc & (allmask ^ (1 << d) ^ adj35[d])
                    for e in range(d + 1, n_old):
                        if (nabcd >> e) & 1:
                            ind5sets.append((a, b, c, d, e))
    
    # Now search for a subset S of {0,...,34} such that:
    #   1. For every triangle (a,b,c): not (a in S and b in S and c in S)
    #   2. For every ind-5-set (a,b,c,d,e): at least one of a,b,c,d,e in S
    
    # Use backtracking with constraint propagation
    # Represent S as a bitmask
    # We'll use a recursive backtracking that assigns vertices one by one
    
    def backtrack(idx, S, forbidden_full):
        """
        idx: next vertex to consider
        S: current neighborhood mask
        forbidden_full: set of triangles fully contained in S (should be empty at solution)
        Returns a valid S or None
        """
        if time.time() > deadline:
            return None
        
        if idx == n_old:
            # Check all constraints
            # Check triangles: no triangle fully in S
            for (a, b, c) in triangles:
                if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1:
                    return None
            # Check ind-5-sets: each must have at least one vertex in S
            for (a, b, c, d, e) in ind5sets:
                if not ((S >> a) & 1 or (S >> b) & 1 or (S >> c) & 1 or (S >> d) & 1 or (S >> e) & 1):
                    return None
            return S
        
        # Try not including idx
        if backtrack(idx + 1, S, forbidden_full):
            return S
        # Try including idx
        if backtrack(idx + 1, S | (1 << idx), forbidden_full):
            return S | (1 << idx)
        return None
    
    # The naive backtracking above is too slow for 35 vertices.
    # Instead, use a smarter approach: iterate over random subsets and check.
    
    # Strategy 2: Randomized local search on the neighborhood of vertex 35
    seeds = [12345, 67890, 11111, 22222, 33333, 44444, 55555, 66666, 77777, 88888,
             99999, 10001, 20002, 30003, 40004, 50005, 60006, 70007, 80008, 90009]
    
    for seed in seeds:
        if time.time() > deadline:
            return None
        
        random.seed(seed)
        
        for attempt in range(8):
            if time.time() > deadline:
                return None
            
            # Create new graph with 36 vertices
            adj = adj35[:] + [0]
            
            # Decide connections of new vertex (vertex 35) to old vertices
            density = 0.3 + 0.4 * random.random()
            for i in range(n_old):
                if random.random() < density:
                    adj[35] |= (1 << i)
                    adj[i] |= (1 << 35)
            
            # Local search to fix violations
            for step in range(15000):
                if time.time() > deadline:
                    return None
                
                k4 = _k4_found(adj, n_new)
                if k4 is None:
                    ind = _ind6_found(adj, n_new)
                    if ind is None:
                        return adj
                    a, b, c, d, e, f = ind
                    if 35 in (a, b, c, d, e, f):
                        others = [x for x in (a, b, c, d, e, f) if x != 35]
                        v = random.choice(others)
                        adj[35] |= (1 << v)
                        adj[v] |= (1 << 35)
                    else:
                        pairs = [(a,b),(a,c),(a,d),(a,e),(a,f),(b,c),(b,d),(b,e),(b,f),(c,d),(c,e),(c,f),(d,e),(d,f),(e,f)]
                        u, v = random.choice(pairs)
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                else:
                    a, b, c, d = k4
                    if 35 in (a, b, c, d):
                        others = [x for x in (a, b, c, d) if x != 35]
                        v = random.choice(others)
                        adj[35] &= ~(1 << v)
                        adj[v] &= ~(1 << 35)
                    else:
                        pairs = [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]
                        u, v = random.choice(pairs)
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
    
    return None

def _local_search_from_seed(adj_seed, n_target, deadline, max_steps=30000):
    """Local search starting from a seed graph, trying to reach n_target vertices."""
    if n_target == len(adj_seed):
        adj = adj_seed[:]
        for step in range(max_steps):
            if time.time() > deadline:
                return None
            k4 = _k4_found(adj, n_target)
            if k4 is None:
                ind = _ind6_found(adj, n_target)
                if ind is None:
                    return adj
                a, b, c, d, e, f = ind
                pairs = [(a,b),(a,c),(a,d),(a,e),(a,f),(b,c),(b,d),(b,e),(b,f),(c,d),(c,e),(c,f),(d,e),(d,f),(e,f)]
                u, v = random.choice(pairs)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            else:
                a, b, c, d = k4
                pairs = [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]
                u, v = random.choice(pairs)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        return None
    else:
        # Need to add vertices
        return None

def construct():
    global _DEADLINE
    start = time.time()
    _DEADLINE = start + 110  # safety margin before 150s kill
    
    # First, try to get the known 35-vertex graph
    adj35 = _build_35_vertex_graph()
    if _is_valid(adj35, 35):
        # We have a valid 35-vertex graph, try to extend to 36
        result36 = _try_extend_35_to_36(adj35, _DEADLINE)
        if result36 is not None:
            return {'n': 36, 'edges': _edges_from_adj(result36, 36)}
        # Return the 35-vertex graph
        return {'n': 35, 'edges': _edges_from_adj(adj35, 35)}
    
    # Fallback: try local search for smaller graphs
    for n in range(30, 16, -1):
        if time.time() > _DEADLINE:
            break
        random.seed(1000 + n)
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        for step in range(15000):
            if time.time() > _DEADLINE:
                break
            k4 = _k4_found(adj, n)
            if k4 is None:
                ind = _ind6_found(adj, n)
                if ind is None:
                    return {'n': n, 'edges': _edges_from_adj(adj, n)}
                a, b, c, d, e, f = ind
                pairs = [(a,b),(a,c),(a,d),(a,e),(a,f),(b,c),(b,d),(b,e),(b,f),(c,d),(c,e),(c,f),(d,e),(d,f),(e,f)]
                u, v = random.choice(pairs)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            else:
                a, b, c, d = k4
                pairs = [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]
                u, v = random.choice(pairs)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
    
    # Final fallback: Paley graph on 17 vertices
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
