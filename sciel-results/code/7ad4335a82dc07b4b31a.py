"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

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

    def solve(n, edges, lists):
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Convert to bitmasks for speed
        adj_mask = [0] * n
        for u in range(n):
            m = 0
            for v in adj[u]:
                m |= (1 << v)
            adj_mask[u] = m

        # Order vertices by degree descending (for MRV heuristic)
        # We'll use a dynamic MRV approach with backtracking
        # For efficiency, we precompute a static order and use forward checking

        # Static ordering: highest degree first
        order = sorted(range(n), key=lambda u: -len(adj[u]))

        # For each vertex, precompute which colors are available
        # lists[v] is the list of allowed colors for vertex v

        # Backtracking with forward checking
        colour = [-1] * n

        # We'll use a recursive backtracking solver
        # To avoid recursion depth issues, we'll implement it iteratively
        # or use sys.setrecursionlimit

        import sys
        sys.setrecursionlimit(10000)

        def backtrack(pos, colour):
            if pos == n:
                return True

            v = order[pos]
            # Check which colors are still available for v
            # A color c is available if c is in lists[v] and no neighbor has color c

            # Get neighbors of v
            neighbors = adj[v]
            # Colors taken by neighbors
            taken = set()
            for u in neighbors:
                if colour[u] != -1:
                    taken.add(colour[u])

            # Try colors in lists[v] that are not taken
            for c in lists[v]:
                if c not in taken:
                    colour[v] = c
                    if backtrack(pos + 1, colour):
                        return True
                    colour[v] = -1

            return False

        if backtrack(0, colour):
            return colour
        else:
            # Should not happen for these instances, but return None
            return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        result = solve(n, edges, lists)
        if result is None:
            # Fallback: return invalid coloring (should not happen)
            result = [0] * n
        out.append(result)
    return out
# EVOLVE-BLOCK-END
