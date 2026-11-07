"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    import sys
    sys.setrecursionlimit(10000)
    
    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_csp(n, edges, lists, rng):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Precompute initial domains as lists for faster iteration
        domains = [list(l) for l in lists]
        domain_sets = [set(l) for l in lists]
        
        # Time limit for backtracking
        import time
        start_time_csp = time.time()
        timeout = 5.0 # 5 seconds per instance attempt
        
        # MRV heuristic: pick unassigned vertex with smallest domain
        def get_mrv_vertex(coloring):
            min_size = float('inf')
            mrv_vertex = -1
            for v in range(n):
                if coloring[v] is None:
                    # Compute current domain considering assigned neighbors
                    taken = set()
                    for u in adj[v]:
                        if coloring[u] is not None:
                            taken.add(coloring[u])
                    current_domain = domain_sets[v] - taken
                    size = len(current_domain)
                    if size < min_size:
                        min_size = size
                        mrv_vertex = v
                    if min_size == 0:
                        break
            return mrv_vertex, min_size

        coloring = [None] * n
        time_exceeded = [False]
        
        def backtrack(depth):
            if time_exceeded[0]:
                return False
            if time.time() - start_time_csp > timeout:
                time_exceeded[0] = True
                return False
                
            if depth == n:
                return True
            
            v, size = get_mrv_vertex(coloring)
            if v == -1:
                return True
            
            if size == 0:
                return False
            
            # Try colors in MRV order
            taken = set()
            for u in adj[v]:
                if coloring[u] is not None:
                    taken.add(coloring[u])
            
            candidates = list(domain_sets[v] - taken)
            rng.shuffle(candidates)
            
            for c in candidates:
                coloring[v] = c
                if backtrack(depth + 1):
                    return True
                coloring[v] = None
            
            return False

        # Try with different random seeds for robustness
        for attempt_seed in range(20):
            rng_attempt = Random(attempt_seed)
            time_exceeded[0] = False
            start_time_csp = time.time()
            if backtrack(0):
                # Verify solution
                valid = True
                for v in range(n):
                    if coloring[v] not in lists[v]:
                        valid = False
                        break
                if valid:
                    for u, v in edges:
                        if coloring[u] == coloring[v]:
                            valid = False
                            break
                if valid:
                    return coloring
            coloring = [None] * n
        
        # Fallback to greedy if CSP fails
        def greedy(n, edges, palette):
            adj_local = [set() for _ in range(n)]
            for u, v in edges:
                adj_local[u].add(v)
                adj_local[v].add(u)
            colour = [None] * n
            for v in sorted(range(n), key=lambda u: -len(adj_local[u])):
                taken = {colour[u] for u in adj_local[v]}
                free = [c for c in palette(v) if c not in taken]
                colour[v] = free[0] if free else palette(v)[0]
            return colour

        return greedy(n, edges, lambda v: lists[v])

    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            break
        n, edges, lists = instance(splits, seed)
        rng = Random(seed)
        out.append(solve_csp(n, edges, lists, rng))
    
    # Ensure we have exactly 4 results
    while len(out) < 4:
        splits, seed = ((20, 701), (40, 702), (60, 703), (100, 704))[len(out)]
        n, edges, lists = instance(splits, seed)
        rng = Random(seed)
        out.append(solve_csp(n, edges, lists, rng))
    
    # Convert to ensure proper types
    result = []
    for coloring in out:
        result.append([int(c) for c in coloring])
    
    return result
# EVOLVE-BLOCK-END
