"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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

    def solve_csp(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Initialize domains
        domains = [set(lst) for lst in lists]
        assigned = [False] * n
        color = [-1] * n
        
        # MRV: select unassigned vertex with smallest domain
        def get_mrv_vertex():
            min_size = float('inf')
            best_v = -1
            for v in range(n):
                if not assigned[v]:
                    size = len(domains[v])
                    if size == 0:
                        return -1 # Contradiction
                    if size < min_size:
                        min_size = size
                        best_v = v
            return best_v

        def backtrack():
            v = get_mrv_vertex()
            if v == -1:
                # Check if all assigned
                for i in range(n):
                    if not assigned[i]:
                        return False
                return True

            # Try each color in domain
            # We iterate over a copy because domains might be modified
            for c in list(domains[v]):
                assigned[v] = True
                color[v] = c
                
                # Forward Check: remove c from neighbors
                changes = []
                consistent = True
                for u in adj[v]:
                    if not assigned[u] and c in domains[u]:
                        domains[u].remove(c)
                        changes.append(u)
                        if len(domains[u]) == 0:
                            consistent = False
                            break
                
                if consistent:
                    if backtrack():
                        return True
                
                # Undo
                for u in changes:
                    domains[u].add(c)
                assigned[v] = False
                color[v] = -1

            return False

        if backtrack():
            return color
        else:
            return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = solve_csp(n, edges, lists)
        
        # Fallback greedy if CSP fails (should not happen)
        if coloring is None:
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            coloring = [0] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = {coloring[u] for u in adj[v]}
                free = [c for c in lists[v] if c not in taken]
                coloring[v] = free[0] if free else lists[v][0]

        out.append([int(c) for c in coloring])
    return out
# EVOLVE-BLOCK-END
