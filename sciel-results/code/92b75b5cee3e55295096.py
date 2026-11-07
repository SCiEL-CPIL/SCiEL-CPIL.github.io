import random, time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(a, b):
    return (a, b) if a < b else (b, a)

def construct():
    t0 = time.time()
    deadline = t0 + 55.0
    n = P['n']
    full_edges = set(_edge_key(u, v) for u, v in P['edges'])
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute degrees for Warnsdorff heuristic
    orig_deg = [len(adj[i]) for i in range(n)]

    def find_hamiltonian_cycle(adj_cur, start, deadline):
        # randomized DFS with backtracking and time check
        n2 = len(adj_cur)
        visited = [False] * n2
        path = [start]
        visited[start] = True
        
        def dfs():
            if time.time() > deadline:
                return False
            if len(path) == n2:
                return start in adj_cur[path[-1]]
            u = path[-1]
            nbrs = list(adj_cur[u])
            # Warnsdorff: prefer neighbors with fewer remaining options
            nbrs.sort(key=lambda w: len(adj_cur[w]))
            random.shuffle(nbrs)
            for w in nbrs:
                if not visited[w]:
                    visited[w] = True
                    path.append(w)
                    if dfs():
                        return True
                    path.pop()
                    visited[w] = False
            return False
        
        if dfs():
            return list(path)
        return None

    def validate_cycles(cycles):
        """Check all cycles are valid and edge-disjoint"""
        used = set()
        for cyc in cycles:
            if len(cyc) < 3:
                return False
            # Check all vertices distinct
            if len(set(cyc)) != len(cyc):
                return False
            # Check all vertices are in [0, n)
            if any(v < 0 or v >= n for v in cyc):
                return False
            for i in range(len(cyc)):
                a, b = cyc[i], cyc[(i + 1) % len(cyc)]
                k = _edge_key(a, b)
                if k not in full_edges:
                    return False
                if k in used:
                    return False
                used.add(k)
        return True

    best = None
    best_count = 0
    
    def try_find_cycles(adj_cur, max_cycles, deadline):
        """Try to find up to max_cycles edge-disjoint Hamiltonian cycles"""
        cycles = []
        for c in range(max_cycles):
            if time.time() > deadline:
                break
            # Pick start vertex: try different strategies
            starts = list(range(n))
            # Sort by current degree (prefer higher degree vertices)
            starts.sort(key=lambda v: -len(adj_cur[v]))
            
            # Try multiple random starts
            found = None
            for trial in range(min(5, len(starts))):
                start = starts[trial]
                cyc = find_hamiltonian_cycle(adj_cur, start, deadline)
                if cyc is not None:
                    found = cyc
                    break
                if time.time() > deadline:
                    break
            
            if found is None:
                break
            
            cycles.append(found)
            
            # Remove edges of this cycle
            for i in range(len(found)):
                a, b = found[i], found[(i + 1) % len(found)]
                adj_cur[a].discard(b)
                adj_cur[b].discard(a)
            
            # Check if graph is still connected enough
            # Quick check: each vertex should have degree >= 2
            if any(len(adj_cur[v]) < 2 for v in range(n)):
                break
        
        return cycles

    rng = random.Random(42)
    attempt = 0
    
    while time.time() < deadline - 2.0:  # Leave 2s margin for final validation
        attempt += 1
        rng = random.Random(attempt * 7919 + 13)
        
        # Work on a copy of adjacency
        adj_cur = [set(s) for s in adj]
        
        # Try to find as many cycles as possible (up to 10)
        cycles = try_find_cycles(adj_cur, 10, deadline - 1.0)
        
        if len(cycles) > 0:
            # Validate
            if validate_cycles(cycles):
                if len(cycles) > best_count:
                    best_count = len(cycles)
                    best = [[int(v) for v in cyc] for cyc in cycles]
                    # If we found 10 cycles, that's likely optimal, return early
                    if best_count >= 10:
                        return best
        
        # If we haven't found anything yet, try a simpler approach
        if best_count == 0 and attempt % 10 == 0:
            # Try greedy: repeatedly find any Hamiltonian cycle
            adj_cur2 = [set(s) for s in adj]
            cycles2 = []
            for c in range(3):
                if time.time() > deadline - 1.0:
                    break
                # Find a Hamiltonian cycle with random restarts
                found = None
                for trial in range(20):
                    start = rng.randrange(n)
                    cyc = find_hamiltonian_cycle(adj_cur2, start, deadline - 0.5)
                    if cyc is not None:
                        found = cyc
                        break
                    if time.time() > deadline - 1.0:
                        break
                
                if found is None:
                    break
                
                cycles2.append(found)
                
                # Remove edges
                for i in range(len(found)):
                    a, b = found[i], found[(i + 1) % len(found)]
                    adj_cur2[a].discard(b)
                    adj_cur2[b].discard(a)
            
            if len(cycles2) > 0 and validate_cycles(cycles2):
                if len(cycles2) > best_count:
                    best_count = len(cycles2)
                    best = [[int(v) for v in cyc] for cyc in cycles2]

    if best is not None:
        return best
    
    # Final fallback: try to find at least one cycle
    adj_cur = [set(s) for s in adj]
    for start in range(n):
        cyc = find_hamiltonian_cycle(adj_cur, start, deadline - 0.1)
        if cyc is not None:
            if validate_cycles([cyc]):
                return [[int(v) for v in cyc]]
    
    # Absolute fallback (should not happen)
    return [[int(v) for v in range(n)]]
