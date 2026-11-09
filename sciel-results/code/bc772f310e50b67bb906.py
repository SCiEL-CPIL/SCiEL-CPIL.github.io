"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import heapq

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

    def solve(n, edges, lists):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Initial domains: each vertex can take any color from its list
        domains = [set(lst) for lst in lists]
        
        # Precompute neighbor lists sorted by degree (higher degree neighbors first for pruning)
        deg = [len(adj[i]) for i in range(n)]
        
        def backtrack(assigned, domains):
            # Find unassigned vertex with minimum remaining values (MRV)
            # Tie-break by higher degree
            best_v = -1
            best_count = None
            best_deg = -1
            for v in range(n):
                if assigned[v] == -1:
                    cnt = len(domains[v])
                    if cnt == 0:
                        return None  # Dead end
                    if best_count is None or cnt < best_count or (cnt == best_count and deg[v] > best_deg):
                        best_v = v
                        best_count = cnt
                        best_deg = deg[v]
            
            if best_v == -1:
                # All assigned
                return list(assigned)
            
            # Try each color in domain of best_v
            for c in sorted(domains[best_v]):
                # Check if c is compatible with already-assigned neighbors
                conflict = False
                for nb in adj[best_v]:
                    if assigned[nb] == c:
                        conflict = True
                        break
                if conflict:
                    continue
                
                # Forward checking: copy domains and prune
                new_domains = [d.copy() for d in domains]
                new_assigned = assigned[:]
                new_assigned[best_v] = c
                
                ok = True
                for nb in adj[best_v]:
                    if new_assigned[nb] == -1:
                        if c in new_domains[nb]:
                            new_domains[nb].discard(c)
                            if len(new_domains[nb]) == 0:
                                ok = False
                                break
                
                if not ok:
                    continue
                
                result = backtrack(new_assigned, new_domains)
                if result is not None:
                    return result
            
            return None
        
        result = backtrack([-1] * n, domains)
        if result is None:
            # Fallback: shouldn't happen by Alon-Tarsi, but return something valid-length
            return [lists[v][0] for v in range(n)]
        return result

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = solve(n, edges, lists)
        # Ensure types are Python int
        coloring = [int(c) for c in coloring]
        out.append(coloring)
    return out
# EVOLVE-BLOCK-END
