"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

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

    def solve_csp(n, edges, color_lists):
        # Build adjacency lists
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        for i in range(n):
            adj[i].sort()

        colour = [-1] * n

        # Use iterative deepening or simple backtracking with MRV
        # For these small graphs, simple backtracking with forward checking is fast enough

        def backtrack(unassigned, domains):
            if not unassigned:
                return True

            # MRV: pick unassigned vertex with fewest available colors
            min_v = -1
            min_count = float('inf')
            for v in unassigned:
                cnt = len(domains[v])
                if cnt < min_count:
                    min_count = cnt
                    min_v = v
                    if cnt <= 1:
                        break
            if min_v == -1:
                return True

            # Try each color for min_v
            for c in domains[min_v]:
                colour[min_v] = c
                # Check consistency with already-colored neighbors
                valid = True
                for nb in adj[min_v]:
                    if colour[nb] == c:
                        valid = False
                        break
                if not valid:
                    continue

                # Forward checking: remove c from unassigned neighbors' domains
                new_domains = None
                failed = False
                for nb in adj[min_v]:
                    if colour[nb] == -1:
                        if new_domains is None:
                            new_domains = dict(domains)
                        if c in new_domains[nb]:
                            new_domains[nb] = [x for x in new_domains[nb] if x != c]
                            if not new_domains[nb]:
                                failed = True
                                break
                if new_domains is None:
                    new_domains = dict(domains)

                if failed:
                    colour[min_v] = -1
                    continue

                new_unassigned = [v for v in unassigned if v != min_v]
                if backtrack(new_unassigned, new_domains):
                    return True

                colour[min_v] = -1

            return False

        # Initialize domains
        domains = [list(color_lists[v]) for v in range(n)]
        unassigned = list(range(n))

        # Order vertices by degree descending for initial ordering (MRV will refine)
        # Actually MRV handles this dynamically

        if backtrack(unassigned, domains):
            return colour
        else:
            # Should not happen per Alon-Tarsi, but return best effort
            return colour

    deadline = time.time() + 140.0  # Safety margin before 150s

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # Fallback: return empty lists (will score 0 for remaining, but at least valid format)
            # Actually we must return 4 lists. Let's try a quick greedy for remaining.
            n, edges, lists = instance(splits, seed)
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = [0] * n
            for v in range(n):
                taken = {colour[u] for u in adj[v]}
                free = [c for c in lists[v] if c not in taken]
                colour[v] = free[0] if free else lists[v][0]
            out.append([int(c) for c in colour])
            continue

        n, edges, lists = instance(splits, seed)
        colour = solve_csp(n, edges, lists)
        # Ensure all are Python ints
        colour = [int(c) for c in colour]
        out.append(colour)

    return out
# EVOLVE-BLOCK-END
