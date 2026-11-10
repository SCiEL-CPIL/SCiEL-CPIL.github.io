"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    
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

    def backtrack_color(n, edges, lists):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Static ordering: highest degree first
        order = sorted(range(n), key=lambda v: -len(adj[v]))
        
        assignment = [None] * n
        domains = [list(lists[i]) for i in range(n)]
        
        def search(idx):
            if idx == n:
                return True
            
            v = order[idx]
            # Try colors in domain
            for c in list(domains[v]):
                if c not in domains[v]:
                    continue
                
                # Save state for undo
                old_domain_v = list(domains[v])
                removed_from_neighbors = []
                
                # Assign
                domains[v].remove(c)
                assignment[v] = c
                
                # Forward check: remove c from neighbors
                feasible = True
                for nb in adj[v]:
                    if c in domains[nb]:
                        domains[nb].remove(c)
                        removed_from_neighbors.append(nb)
                        if not domains[nb]:
                            feasible = False
                            break
                
                if feasible:
                    if search(idx + 1):
                        return True
                
                # Undo
                domains[v] = old_domain_v
                assignment[v] = None
                for nb in removed_from_neighbors:
                    domains[nb].append(c)
                    domains[nb].sort()
            
            return False
        
        if search(0):
            return assignment
        else:
            return [lists[i][0] for i in range(n)]
    
    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = backtrack_color(n, edges, lists)
        out.append([int(c) for c in coloring])
    return out
# EVOLVE-BLOCK-END
