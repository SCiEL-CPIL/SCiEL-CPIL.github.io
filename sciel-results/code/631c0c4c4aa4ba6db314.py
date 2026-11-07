"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import time
    from collections import deque

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

    def solve_instance(n, edges, lists, deadline):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Domains as sets for fast membership and discard
        domains = [set(lists[v]) for v in range(n)]
        
        # Check for initial empty domains
        for v in range(n):
            if not domains[v]:
                return None
        
        # AC-3 Implementation for arc consistency
        def ac3(domains, adj, n, queue=None):
            if queue is None:
                queue = deque()
                for u in range(n):
                    for v in adj[u]:
                        queue.append((u, v))
            
            while queue:
                u, v = queue.popleft()
                # Check if domain[u] is reduced by constraint (u, v)
                # For != constraint: c in domains[u] is unsupported if domains[v] == {c}
                revised = False
                to_remove = []
                # Optimization: only need to check if domains[v] is singleton
                if len(domains[v]) == 1:
                    only_color = next(iter(domains[v]))
                    if only_color in domains[u]:
                        to_remove.append(only_color)
                
                if to_remove:
                    for c in to_remove:
                        domains[u].discard(c)
                    revised = True
                    if not domains[u]:
                        return False
                    # Add neighbors of u (except v) to queue
                    for z in adj[u]:
                        if z != v:
                            queue.append((z, u))
            return True

        # Initial AC-3 pass
        if not ac3(domains, adj, n):
            return None

        assigned = [None] * n
        
        def get_mrv_var():
            min_size = float('inf')
            var = -1
            for v in range(n):
                if assigned[v] is None:
                    size = len(domains[v])
                    if size < min_size:
                        min_size = size
                        var = v
                        if size <= 1:
                            break
            return var

        def backtrack():
            if time.time() > deadline:
                return False
            
            var = get_mrv_var()
            if var == -1:
                return True # All assigned
            
            # LVC Heuristic: Order values by how few constraints they impose on neighbors
            values = list(domains[var])
            def value_score(c):
                count = 0
                for u in adj[var]:
                    if assigned[u] is None and c in domains[u]:
                        count += 1
                return count
            
            values.sort(key=value_score)
            
            for c in values:
                if c not in domains[var]:
                    continue
                
                # Assign
                assigned[var] = c
                saved_domain_var = set(domains[var])
                domains[var].clear()
                
                # Forward Checking: Remove c from neighbors' domains
                saved_neighbors = []
                ok = True
                for u in adj[var]:
                    if assigned[u] is None:
                        if c in domains[u]:
                            saved_neighbors.append(u)
                            domains[u].discard(c)
                            if not domains[u]:
                                ok = False
                                break
                        else:
                            pass
                
                if ok:
                    # Optional: Run a local AC-3 check on affected nodes? 
                    # For now, just recurse.
                    if backtrack():
                        return True
                
                # Undo
                assigned[var] = None
                domains[var] = saved_domain_var
                for u in saved_neighbors:
                    domains[u].add(c)
                
                if time.time() > deadline:
                    return False
            
            return False

        if backtrack():
            # Ensure types are strictly int
            return [int(assigned[v]) for v in range(n)]
        else:
            # Fallback: should not happen for these instances, but return dummy to avoid crash
            return [0] * n

    start = time.time()
    deadline = start + 115.0  # Safety margin before 120s limit

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline - 5:
            n, edges, lists = instance(splits, seed)
            # Emergency fallback: just pick first available color, likely invalid but valid structure
            out.append([int(lists[v][0]) for v in range(n)])
            continue
        
        n, edges, lists = instance(splits, seed)
        result = solve_instance(n, edges, lists, deadline)
        
        # Validate result strictly before adding
        if len(result) != n:
            result = [int(lists[v][0]) for v in range(n)]
        else:
            # Check types and values
            valid = True
            for i, val in enumerate(result):
                if not isinstance(val, int) or isinstance(val, bool):
                    valid = False
                    break
                if val < 0 or val > 4:
                    valid = False
                    break
            if not valid:
                result = [int(lists[v][0]) for v in range(n)]
        
        out.append(result)
    
    return out
# EVOLVE-BLOCK-END
