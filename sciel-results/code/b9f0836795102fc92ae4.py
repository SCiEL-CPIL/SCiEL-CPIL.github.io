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

    def color_instance(n, edges, lists):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # We need to assign a color from lists[v] to each vertex v such that
        # adjacent vertices have different colors.
        # Since the graph is bipartite and planar, and lists have size 3,
        # it is always colorable. We use a simple backtracking with heuristics.
        # However, for efficiency on up to 108 vertices, we can try a greedy
        # approach with randomization and restarts, or a more structured approach.
        
        # Let's use a deterministic backtracking with MRV (Minimum Remaining Values)
        # and forward checking. But since it's always solvable, a simple greedy
        # with proper ordering might work. Let's try a randomized greedy first,
        # and if it fails, fall back to backtracking.
        
        # Actually, let's use a proper CSP solver with backtracking and MRV.
        # Given the small size, this should be fast.
        
        import sys
        sys.setrecursionlimit(10000)
        
        # Precompute neighbors
        neighbors = adj
        
        # Order vertices by degree (descending) for initial assignment
        order = sorted(range(n), key=lambda v: -len(neighbors[v]))
        
        # Try greedy first
        def try_greedy(order):
            color = [None] * n
            for v in order:
                taken = set()
                for u in neighbors[v]:
                    if color[u] is not None:
                        taken.add(color[u])
                available = [c for c in lists[v] if c not in taken]
                if not available:
                    return None
                color[v] = available[0]
            # Verify
            for u in range(n):
                for v in neighbors[u]:
                    if u < v and color[u] == color[v]:
                        return None
            return color
        
        result = try_greedy(order)
        if result is not None:
            return result
        
        # Try randomized greedy with multiple restarts
        import random
        rng = Random(42)
        for attempt in range(100):
            color = [None] * n
            success = True
            # Shuffle order slightly
            o = order[:]
            rng.shuffle(o)
            for v in o:
                taken = set()
                for u in neighbors[v]:
                    if color[u] is not None:
                        taken.add(color[u])
                available = [c for c in lists[v] if c not in taken]
                if not available:
                    success = False
                    break
                color[v] = rng.choice(available)
            if success:
                # Verify
                valid = True
                for u in range(n):
                    for v in neighbors[u]:
                        if u < v and color[u] == color[v]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return color
        
        # Fall back to backtracking with MRV
        # This is a proper CSP solver
        def backtrack():
            color = [None] * n
            
            # MRV: pick uncolored vertex with fewest available colors
            def select_variable():
                best_v = None
                best_count = float('inf')
                for v in range(n):
                    if color[v] is None:
                        taken = set()
                        for u in neighbors[v]:
                            if color[u] is not None:
                                taken.add(color[u])
                        avail = [c for c in lists[v] if c not in taken]
                        if len(avail) < best_count:
                            best_count = len(avail)
                            best_v = v
                            if best_count == 0:
                                return v, []
                return best_v, [c for c in lists[best_v] if c not in {color[u] for u in neighbors[best_v] if color[u] is not None}] if best_v is not None else (None, [])
            
            def rec():
                v, avail = select_variable()
                if v is None:
                    return True  # All colored
                if not avail:
                    return False
                for c in avail:
                    color[v] = c
                    if rec():
                        return True
                    color[v] = None
                return False
            
            if rec():
                return color
            return None
        
        result = backtrack()
        if result is not None:
            return result
        
        # This should never happen as the instance is always colorable
        raise Exception("Failed to color instance")

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = color_instance(n, edges, lists)
        # Ensure all values are Python ints
        coloring = [int(c) for c in coloring]
        out.append(coloring)
    return out
# EVOLVE-BLOCK-END
