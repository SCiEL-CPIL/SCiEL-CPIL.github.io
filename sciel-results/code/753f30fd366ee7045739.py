"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from random import Random
    
    deadline = time.time() + 110.0 # Safety margin before 120s limit

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

    def solve_list_coloring(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Initial domains
        domains = [set(lst) for lst in lists]
        color = [-1] * n
        
        # MRV + Forward Checking Backtracking
        def backtrack(depth):
            if time.time() > deadline:
                return False
                
            if depth == n:
                return True
                
            # Find uncolored vertex with Minimum Remaining Values (MRV)
            min_domain_size = float('inf')
            min_vertex = -1
            for i in range(n):
                if color[i] == -1:
                    if len(domains[i]) < min_domain_size:
                        min_domain_size = len(domains[i])
                        min_vertex = i
                        if min_domain_size == 0:
                            return False
                        if min_domain_size == 1:
                            break # Can't get better than 1
                
            if min_vertex == -1:
                return True # All colored

            # Try colors for the selected vertex
            # Sort colors by least constraining value (LCV) heuristic is optional, 
            # but for small domains, just iterating is fine.
            # We iterate over a copy because we might modify domains? No, we restore.
            for c in sorted(domains[min_vertex]):
                # Check if color c is valid for neighbors
                valid = True
                for nb in adj[min_vertex]:
                    if color[nb] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Assign
                color[min_vertex] = c
                saved_domains = [None] * n
                # Propagate: remove c from neighbors' domains
                for nb in adj[min_vertex]:
                    if color[nb] == -1:
                        saved_domains[nb] = set(domains[nb])
                        if c in domains[nb]:
                            domains[nb].remove(c)
                            if len(domains[nb]) == 0:
                                # Conflict! Restore and break
                                for j in range(n):
                                    if saved_domains[j] is not None:
                                        domains[j] = saved_domains[j]
                                color[min_vertex] = -1
                                valid = False
                                break
                
                if valid:
                    if backtrack(depth + 1):
                        return True
                    else:
                        # Restore domains if backtrack failed
                        for j in range(n):
                            if saved_domains[j] is not None:
                                domains[j] = saved_domains[j]
                
                color[min_vertex] = -1
                
            return False

        if backtrack(0):
            return color
        else:
            return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # Fallback to greedy if we run out of time, though unlikely
            def greedy(n, edges, palette):
                adj = [set() for _ in range(n)]
                for u, v in edges:
                    adj[u].add(v)
                    adj[v].add(u)
                colour = [None] * n
                for v in sorted(range(n), key=lambda u: -len(adj[u])):
                    taken = {colour[u] for u in adj[v] if colour[u] is not None}
                    free = [c for c in palette(v) if c not in taken]
                    colour[v] = free[0] if free else palette(v)[0]
                return colour
            n, edges, lists = instance(splits, seed)
            out.append(greedy(n, edges, lambda v: lists[v]))
            continue
            
        n, edges, lists = instance(splits, seed)
        sol = solve_list_coloring(n, edges, lists)
        if sol is None:
             # Should not happen due to Alon-Tarsi, but handle gracefully
             # Fallback to greedy
             adj = [set() for _ in range(n)]
             for u, v in edges:
                 adj[u].add(v)
                 adj[v].add(u)
             colour = [None] * n
             for v in sorted(range(n), key=lambda u: -len(adj[u])):
                 taken = {colour[u] for u in adj[v] if colour[u] is not None}
                 free = [c for c in lists[v] if c not in taken]
                 colour[v] = free[0] if free else lists[v][0]
             out.append(colour)
        else:
             out.append([int(c) for c in sol])
             
    return out
# EVOLVE-BLOCK-END
