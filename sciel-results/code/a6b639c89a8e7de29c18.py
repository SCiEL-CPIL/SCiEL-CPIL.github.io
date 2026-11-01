"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    import time
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin

    def build_luw(q, d):
        """Build LUW(d) graph over GF(q). Returns (n, edges)."""
        # Precompute modular arithmetic
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q
        
        # Define the recurrence rule
        rule = {}
        rule[2] = (1, 1)
        rule[3] = (2, 1)
        rule[4] = (1, 2)
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
        # Generate all vertices
        # Vertex: (side, x) where side in {0,1}, x = (x1, ..., xd) in GF(q)^d
        # Total: 2 * q^d vertices
        n = 2 * q ** d
        edges = set()
        
        # Map vertex to index
        def vertex_to_index(side, x_tuple):
            # x_tuple is a tuple of d elements
            idx = 0
            for i in range(d):
                idx = idx * q + x_tuple[i]
            return side * (q ** d) + idx
        
        # For each vertex on side 0, generate neighbors on side 1
        for x1 in range(q):
            for x2 in range(q):
                # ... this gets complex for large d. Let's use a different approach.
                pass
        
        # Better: iterate over all vertices systematically
        edges_list = []
        seen_edges = set()
        
        for side in [0]:  # Only generate from side 0 to avoid duplicates
            for idx in range(q ** d):
                # Decode idx to tuple
                x = []
                temp = idx
                for i in range(d):
                    x.append(temp % q)
                    temp //= q
                x = tuple(x)
                
                # For each "first" value in GF(q), compute the neighbor
                for first in range(q):
                    # y[0] is "None" (not used), y[1] = first, y[2..d] computed
                    y = [None] + [0] * d  # y[0] unused, y[1..d]
                    y[1] = first
                    for c in range(2, d + 1):
                        a, b = rule[c]
                        # x is 0-indexed: x[0]..x[d-1] correspond to x[1]..x[d]
                        # y is 1-indexed: y[1]..y[d]
                        if side == 0:
                            y[c] = add(x[c-1], mul(y[a], x[b-1]))
                        else:
                            y[c] = sub(x[c-1], mul(x[a-1], y[b]))
                    neighbor_idx = vertex_to_index(1 - side, tuple(y[1:]))
                    v_idx = vertex_to_index(side, x)
                    edge = (v_idx, neighbor_idx)
                    if edge not in seen_edges:
                        seen_edges.add(edge)
                        edges_list.append(edge)
        
        return n, edges_list

    def verify_girth_nonbacktracking(n, edges, target_girth):
        """Verify girth >= target_girth using non-backtracking walks.
        A cycle of length L exists iff there's a non-backtracking closed walk of length L.
        Check all L < target_girth."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Build directed edge list
        dir_edges = []
        dir_edge_idx = {}
        for u in range(n):
            for v in adj[u]:
                key = (u, v)
                if key not in dir_edge_idx:
                    dir_edge_idx[key] = len(dir_edges)
                    dir_edges.append((u, v))
        
        num_dir_edges = len(dir_edges)
        
        # Build non-backtracking transition: from (u,v) go to (v,w) for w != u
        nb_trans = [[] for _ in range(num_dir_edges)]
        for (u, v), idx in dir_edge_idx.items():
            for w in adj[v]:
                if w != u:
                    next_key = (v, w)
                    if next_key in dir_edge_idx:
                        nb_trans[idx].append(dir_edge_idx[next_key])
        
        # For each L from 2 to target_girth-1, check if any non-backtracking
        # closed walk of length L exists.
        # A non-backtracking closed walk of length L: v0->v1->...->vL=v0
        # In directed edge terms: (v0,v1), (v1,v2), ..., (v_{L-1},v0)
        # This is a path of length L-1 in the NB graph from (v0,v1) to (v_{L-1},v0)
        # where the last edge is (v_{L-1}, v0) and we need v0 = start vertex.
        
        # More precisely: a cycle of length L corresponds to a sequence of L directed edges
        # e1=(v0,v1), e2=(v1,v2), ..., eL=(v_{L-1},v0)
        # In the NB graph, this is a walk of length L-1 from e1 to eL.
        # We need: the "end" of eL is v0, and the "start" of e1 is v0.
        # i.e., eL = (v_{L-1}, v0) and e1 = (v0, v1), so end(eL) = start(e1).
        
        # So: for some starting directed edge e1=(a,b), is there a path of length L-1
        # in the NB graph from e1 to some edge eL=(c,a) where end(eL)=a=start(e1)?
        
        # Equivalently: for each directed edge e1=(a,b), does the NB graph have a path
        # of length L-1 from e1 to any edge ending at a?
        
        # For small L, we can do iterative walks.
        # State after t steps: set of directed edges reachable.
        # After L-1 steps from e1, check if any edge in the set ends at start(e1).
        
        for L in range(2, target_girth):
            # For each starting directed edge, walk L-1 steps and check closure
            # This is O(num_dir_edges * (avg_degree-1)^(L-1)) which can be expensive
            # But for L up to 8 and num_dir_edges ~ 2*n*k = 12500, it's manageable
            # if we do it smartly.
            
            # Alternative: use the fact that we only need to check if ANY closed walk exists.
            # We can do a simultaneous walk from all starting edges.
            # State: dict mapping directed_edge_idx -> set of starting_edges that can reach it
            # But that's expensive.
            
            # Better: for each starting edge, do a DFS/BFS of depth L-1.
            # With L-1 <= 7 and branching factor ~ k-1 = 4, max nodes = 4^7 = 16384 per start.
            # Total: 12500 * 16384 = 204M which is too much.
            
            # Even better: use the matrix approach or iterative sets.
            # Let's use iterative set propagation.
            
            # After 0 steps: each edge e is reachable from itself.
            # After 1 step: edge f is reachable from e if f is in nb_trans[e].
            # After t steps: edges reachable from e after t steps.
            
            # We want: for some e=(a,b), is there an edge f=(c,a) reachable from e in L-1 steps?
            
            # Let's do: for each L, maintain a dict: edge_idx -> set of start_vertices
            # that can reach this edge in the current number of steps.
            # Initially (0 steps): edge (a,b) -> {a}
            # After 1 step: for each edge e=(a,b) in nb_trans, add a to edge's set.
            # After L-1 steps: check if any edge (c,a) has a in its set.
            
            # This is O(L * num_dir_edges * avg_degree) which is fine.
            
            # Initialize: edge_reachable[edge_idx] = {start_vertex_of_edge}
            edge_reachable = {}
            for (u, v), idx in dir_edge_idx.items():
                edge_reachable[idx] = {u}
            
            # Iterate L-1 times
            for step in range(L - 1):
                new_reachable = {}
                for e_idx, starts in edge_reachable.items():
                    for f_idx in nb_trans[e_idx]:
                        if f_idx not in new_reachable:
                            new_reachable[f_idx] = set()
                        new_reachable[f_idx].update(starts)
                edge_reachable = new_reachable
            
            # Check if any edge (c, a) has a in its reachable set
            for (u, v), idx in dir_edge_idx.items():
                # Edge (u,v): end vertex is v. Check if v is in edge_reachable[idx]
                if idx in edge_reachable and v in edge_reachable[idx]:
                    return False  # Found a cycle of length L
        
        return True

    # For k=5, g=9:
    # LUW(d) over GF(q) has degree q and girth >= 2d+2
    # We need degree k, so q = k = 5
    # We need girth >= g, so 2d+2 >= g, i.e., d >= (g-2)/2
    # For g=9: d >= 3.5, so d >= 4
    # LUW(4) over GF(5): n = 2*5^4 = 1250
    
    q = k
    min_d = max(2, (g - 2) // 2 + 1)  # d such that 2d+2 >= g
    
    best_result = None
    
    for d in range(min_d, min_d + 3):
        if time.time() > deadline - 30:
            break
        try:
            n, edges = build_luw(q, d)
            # Verify the graph is k-regular
            deg = [0] * n
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            if any(deg[i] != k for i in range(n)):
                continue
            # Verify girth
            if verify_girth_nonbacktracking(n, edges, g):
                if best_result is None or n < best_result[0]:
                    best_result = (n, edges)
        except Exception as e:
            continue
    
    if best_result is not None:
        return best_result
    
    # Fallback: try larger d
    for d in range(min_d + 3, min_d + 6):
        if time.time() > deadline:
            break
        try:
            n, edges = build_luw(q, d)
            deg = [0] * n
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            if any(deg[i] != k for i in range(n)):
                continue
            if verify_girth_nonbacktracking(n, edges, g):
                if best_result is None or n < best_result[0]:
                    best_result = (n, edges)
        except Exception:
            continue
    
    if best_result is not None:
        return best_result
    
    # Last resort
    d = min_d + 2
    n, edges = build_luw(q, d)
    return n, edges
# EVOLVE-BLOCK-END
