"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    import sys
    sys.setrecursionlimit(100000)

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
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Degree for ordering
        deg = [len(adj[v]) for v in range(n)]

        # Colour assignment: -1 means uncoloured
        colour = [-1] * n
        # Domains: list of available colours per vertex (from its list)
        domains = [list(lists[v]) for v in range(n)]

        # Count how many neighbours each vertex has in the current "frontier"
        # We'll use a simple backtracking with MRV (minimum remaining values) heuristic

        # For efficiency, track uncoloured vertices and use a priority approach
        # But since n is small (up to ~408), we can do a straightforward backtracking

        # Sort vertices by degree descending as initial ordering, but use MRV dynamically
        # Let's use a recursive backtracking with MRV

        # To make it faster, pre-compute some structures
        # Use a list of uncoloured vertices
        uncoloured = set(range(n))

        # We'll use a simple recursive backtracking
        # With n up to ~408, and 3 colours each, this should be fast enough

        def backtrack():
            if not uncoloured:
                return True

            # MRV: pick vertex with fewest available colours
            best_v = None
            best_count = 10
            for v in uncoloured:
                # Count available colours (not used by coloured neighbours)
                taken = set()
                for u in adj[v]:
                    if colour[u] != -1:
                        taken.add(colour[u])
                avail = [c for c in domains[v] if c not in taken]
                cnt = len(avail)
                if cnt == 0:
                    return False
                if cnt < best_count:
                    best_count = cnt
                    best_v = v
                    if cnt == 1:
                        break

            if best_v is None:
                return False

            v = best_v
            taken = set()
            for u in adj[v]:
                if colour[u] != -1:
                    taken.add(colour[u])
            avail = [c for c in domains[v] if c not in taken]

            # Try each available colour
            for c in avail:
                colour[v] = c
                uncoloured.remove(v)
                if backtrack():
                    return True
                uncoloured.add(v)
                colour[v] = -1

            return False

        if backtrack():
            return [colour[v] for v in range(n)]
        else:
            # Should not happen per Alon-Tarsi, but return something
            return [domains[v][0] for v in range(n)]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        out.append(solve_csp(n, edges, lists))

    # Ensure types are correct: Python int, list of lists
    result = []
    for colouring in out:
        result.append([int(c) for c in colouring])
    return result
# EVOLVE-BLOCK-END
