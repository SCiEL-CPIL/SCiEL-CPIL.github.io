import time
import random

def _build_adj(n, edges):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _has_k4(adj, n):
    """Check if graph has a K4. Return True if found."""
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                common = adj[u] & adj[v]
                m = common
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    m ^= lsb
                    if common & adj[w]:
                        return True
    return False

def _has_independent_6(adj, n):
    """Check if graph has an independent set of size 6."""
    full_mask = (1 << n) - 1
    comp_adj = [0] * n
    for i in range(n):
        comp_adj[i] = full_mask ^ adj[i] ^ (1 << i)
    
    def dfs(verts, depth):
        if depth == 6:
            return True
        if not verts:
            return False
        v = verts
        while v:
            lsb = v & -v
            u = lsb.bit_length() - 1
            v ^= lsb
            new_verts = v & comp_adj[u]
            if dfs(new_verts, depth + 1):
                return True
        return False
    return dfs(full_mask, 0)

def _is_valid_graph(n, edges):
    adj = _build_adj(n, edges)
    if _has_k4(adj, n):
        return False
    if _has_independent_6(adj, n):
        return False
    return True

def _edges_from_adj(adj, n):
    edges = []
    for u in range(n):
        m = adj[u]
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            if u < v:
                edges.append([u, v])
    return edges

def _paley_graph(p):
    """Generate Paley graph for prime p."""
    squares = {x * x % p for x in range(1, p)}
    edges = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                edges.append([i, j])
    return {'n': p, 'edges': edges}

def _clebsch_graph():
    """Generate the Clebsch graph: 16 vertices, K4-free, alpha=5."""
    n = 16
    vertices = []
    for i in range(16):
        bits = bin(i).count('1')
        if bits % 2 == 0:
            vertices.append(i)
    idx_map = {v: i for i, v in enumerate(vertices)}
    edges = []
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            diff = bin(vertices[i] ^ vertices[j]).count('1')
            if diff in (1, 3):
                edges.append([i, j])
    return {'n': n, 'edges': edges}

def _constraint_backtrack(n, time_limit, seed=None):
    """
    Constraint-programming-style backtracking search.
    Builds graph vertex-by-vertex. For each new vertex v (index i),
    choose its adjacency to vertices 0..i-1 from safe patterns.
    
    A pattern is "safe" if:
    1. It doesn't create a K4: for any two neighbors u,w of v that are adjacent,
       u and w must not have a common neighbor x that is also a neighbor of v.
       More precisely: for any triple (u,w,x) where u~w, u~x, w~x (triangle),
       v cannot be adjacent to all three.
       Equivalently: the neighbors of v must form a K3-free subgraph.
    2. It doesn't create an independent 6-set: for any 5 vertices that are
       mutually non-adjacent (independent 5-set) AND none of them is adjacent
       to v, adding v would make an independent 6-set.
       So: v must be adjacent to at least one vertex in every independent 5-set.
    """
    if seed is not None:
        random.seed(seed)
    
    start = time.time()
    deadline = start + time_limit
    
    # adj[i] = bitmask of neighbors of i among vertices 0..n-1
    adj = [0] * n
    # For tracking: neighbors[i] = set of neighbors of i
    neighbors = [set() for _ in range(n)]
    
    def is_safe_pattern(i, pattern):
        """Check if connecting vertex i to vertices in 'pattern' (bitmask) is safe."""
        # Check K4 constraint:
        # The neighbors of vertex i (among 0..i-1) must not contain a K3.
        # Because if u,w,x are all neighbors of i and u~w, w~x, u~x, then {i,u,w,x} is K4.
        neigh_mask = pattern
        # Check if the subgraph induced by neighbors of i contains a triangle
        m = neigh_mask
        while m:
            lsb1 = m & -m
            u = lsb1.bit_length() - 1
            m ^= lsb1
            m2 = m
            while m2:
                lsb2 = m2 & -m2
                w = lsb2.bit_length() - 1
                m2 ^= lsb2
                if (neigh_mask >> w) & 1:  # w is also a neighbor of i
                    # Check if u~w
                    if (adj[u] >> w) & 1:
                        # u and w are adjacent and both neighbors of i
                        # Check if there's a common neighbor x of u and w that is also neighbor of i
                        common_uw = adj[u] & adj[w]
                        if common_uw & neigh_mask:
                            return False
        # Check independent 6-set constraint:
        # v=i must be adjacent to at least one vertex in every independent 5-set
        # among vertices 0..i-1.
        # Equivalently: the non-neighbors of i (among 0..i-1) must not contain an independent 5-set.
        non_neigh_mask = ((1 << i) - 1) ^ neigh_mask
        # Check if non_neigh_mask contains an independent 5-set
        # In the complement graph (restricted to 0..i-1), check for K5
        # Actually, independent 5-set in G = K5 in complement
        # Let's check: is there a set of 5 vertices in non_neigh_mask that are pairwise non-adjacent in G?
        def has_independent_5_in(mask, vertices_count):
            """Check if the set of vertices in 'mask' contains an independent 5-set."""
            # Use the complement adjacency for this subgraph
            def dfs(verts, depth):
                if depth == 5:
                    return True
                if not verts:
                    return False
                v = verts
                while v:
                    lsb = v & -v
                    u = lsb.bit_length() - 1
                    v ^= lsb
                    # New candidates: vertices in v that are non-adjacent to u
                    new_verts = 0
                    vv = v
                    while vv:
                        lsb2 = vv & -vv
                        w = lsb2.bit_length() - 1
                        vv ^= lsb2
                        if not ((adj[u] >> w) & 1):
                            new_verts |= (1 << w)
                    if dfs(new_verts, depth + 1):
                        return True
                return False
            return dfs(mask, 0)
        
        if has_independent_5_in(non_neigh_mask, i):
            return False
        
        return True
    
    def backtrack(i):
        """Try to assign adjacency for vertex i to vertices 0..i-1."""
        if time.time() > deadline:
            return False
        if i == n:
            return True
        
        full_prev = (1 << i) - 1
        
        # Generate all possible patterns and filter safe ones
        # For efficiency, we'll use a heuristic: try patterns in order of "balance"
        # Start with medium density patterns
        
        # Collect safe patterns
        safe_patterns = []
        # We can't enumerate all 2^i patterns for large i, so use sampling or heuristics
        if i <= 12:
            # Enumerate all patterns
            for pattern in range(1 << i):
                if is_safe_pattern(i, pattern):
                    safe_patterns.append(pattern)
        else:
            # For larger i, use heuristic: try specific density levels
            # Sample patterns at different densities
            for density in [0.3, 0.4, 0.5, 0.6, 0.7]:
                for _ in range(50):
                    pattern = 0
                    for j in range(i):
                        if random.random() < density:
                            pattern |= (1 << j)
                    if is_safe_pattern(i, pattern):
                        safe_patterns.append(pattern)
            # Also try all-0 and all-1
            if is_safe_pattern(i, 0):
                safe_patterns.append(0)
            if is_safe_pattern(i, full_prev):
                safe_patterns.append(full_prev)
        
        if not safe_patterns:
            return False
        
        # Shuffle for randomization
        random.shuffle(safe_patterns)
        
        for pattern in safe_patterns:
            if time.time() > deadline:
                return False
            # Apply pattern
            adj[i] = pattern
            for j in range(i):
                if (pattern >> j) & 1:
                    adj[j] |= (1 << i)
                    neighbors[j].add(i)
            neighbors[i] = set(j for j in range(i) if (pattern >> j) & 1)
            
            if backtrack(i + 1):
                return True
            
            # Undo
            for j in range(i):
                if (pattern >> j) & 1:
                    adj[j] &= ~(1 << i)
                    neighbors[j].discard(i)
            adj[i] = 0
            neighbors[i] = set()
        
        return False
    
    # Try to build the graph
    if backtrack(0):
        return _edges_from_adj(adj, n)
    return None

def _local_search(n, time_limit, seed=None):
    """Local search to find a valid graph on n vertices (fallback)."""
    if seed is not None:
        random.seed(seed)
    
    start = time.time()
    deadline = start + time_limit
    
    # Start with random graph
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < 0.5:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    best_valid = None
    
    while time.time() < deadline:
        # Check validity
        if not _has_k4(adj, n) and not _has_independent_6(adj, n):
            if n > 0:
                best_valid = adj[:]
            break
        
        # Find a violation and try to fix it
        if _has_k4(adj, n):
            found = False
            for u in range(n):
                if found:
                    break
                for v in range(u + 1, n):
                    if found:
                        break
                    if (adj[u] >> v) & 1:
                        common = adj[u] & adj[v]
                        m = common
                        while m:
                            lsb = m & -m
                            w = lsb.bit_length() - 1
                            m ^= lsb
                            if common & adj[w]:
                                x_candidates = common & adj[w]
                                xc = x_candidates
                                while xc:
                                    lsb2 = xc & -xc
                                    x = lsb2.bit_length() - 1
                                    xc ^= lsb2
                                    k4 = [u, v, w, x]
                                    i, j = random.sample(k4, 2)
                                    adj[i] &= ~(1 << j)
                                    adj[j] &= ~(1 << i)
                                    found = True
                                    break
                            if found:
                                break
                    if found:
                        break
        else:
            full_mask = (1 << n) - 1
            comp_adj = [0] * n
            for i in range(n):
                comp_adj[i] = full_mask ^ adj[i] ^ (1 << i)
            
            def find_independent_6(verts, path):
                if len(path) == 6:
                    return path[:]
                if not verts:
                    return None
                v = verts
                while v:
                    lsb = v & -v
                    u = lsb.bit_length() - 1
                    v ^= lsb
                    new_verts = v & comp_adj[u]
                    result = find_independent_6(new_verts, path + [u])
                    if result:
                        return result
                return None
            
            ind_set = find_independent_6(full_mask, [])
            if ind_set:
                i, j = random.sample(ind_set, 2)
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    if best_valid is not None:
        return _edges_from_adj(best_valid, n)
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # First, try known constructions
    candidates = [
        ('paley37', _paley_graph(37)),
        ('paley29', _paley_graph(29)),
        ('paley17', _paley_graph(17)),
        ('clebsch', _clebsch_graph()),
    ]
    
    best_graph = None
    best_n = 0
    
    for name, g in candidates:
        if time.time() > deadline:
            break
        if _is_valid_graph(g['n'], g['edges']):
            if g['n'] > best_n:
                best_graph = g
                best_n = g['n']
    
    # Now try constraint backtracking for larger n
    for target_n in [35, 34, 33, 32, 30, 28, 26, 24, 22, 20]:
        if time.time() > deadline:
            break
        if target_n <= best_n:
            continue
        
        remaining = deadline - time.time()
        if remaining < 3:
            break
        
        # Give each target a time budget
        time_per_target = min(remaining * 0.6, 20)
        
        found = False
        for attempt in range(3):
            if time.time() > deadline:
                break
            edges = _constraint_backtrack(target_n, time_per_target / 3, seed=100 + attempt)
            if edges is not None:
                g = {'n': target_n, 'edges': edges}
                if _is_valid_graph(target_n, edges):
                    if target_n > best_n:
                        best_graph = g
                        best_n = target_n
                    found = True
                    break
        
        if found:
            # Found a valid graph, try to extend to larger n
            # But first, break to try even larger n in next iteration
            pass
    
    # If backtracking didn't find anything better, try local search as fallback
    if best_n < 35:
        for target_n in [35, 33, 31, 29, 27, 25]:
            if time.time() > deadline:
                break
            if target_n <= best_n:
                continue
            
            remaining = deadline - time.time()
            if remaining < 2:
                break
            time_per_target = min(remaining * 0.5, 10)
            
            for attempt in range(3):
                if time.time() > deadline:
                    break
                edges = _local_search(target_n, time_per_target / 3, seed=200 + attempt)
                if edges is not None:
                    g = {'n': target_n, 'edges': edges}
                    if _is_valid_graph(target_n, edges):
                        if target_n > best_n:
                            best_graph = g
                            best_n = target_n
                        break
    
    if best_graph is None:
        best_graph = _clebsch_graph()
    
    return best_graph
