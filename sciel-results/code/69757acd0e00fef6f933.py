"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import sys
    from random import Random
    from collections import deque

    # Keep the exact generator under the name instance
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
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        # Convert to sets for faster lookup
        adj_set = [set(s) for s in adj]

        # Backtracking with MRV and forward checking
        # colour[v] = -1 means uncoloured
        colour = [-1] * n

        def check_deadline():
            return time.time() < deadline

        def backtrack():
            if not check_deadline():
                return False

            # Find uncoloured vertex with MRV (minimum remaining values)
            best_v = -1
            best_count = float('inf')
            for v in range(n):
                if colour[v] == -1:
                    taken = {colour[u] for u in adj[v] if colour[u] != -1}
                    available = [c for c in lists[v] if c not in taken]
                    if len(available) < best_count:
                        best_count = len(available)
                        best_v = v
                        if best_count == 0:
                            return False
                    if best_count == 1:
                        break

            if best_v == -1:
                return True  # All coloured

            taken = {colour[u] for u in adj[best_v] if colour[u] != -1}
            available = [c for c in lists[best_v] if c not in taken]

            for c in available:
                if not check_deadline():
                    return False
                colour[best_v] = c
                # Forward checking: verify all uncoloured neighbours still have options
                valid = True
                for u in adj[best_v]:
                    if colour[u] == -1:
                        taken_u = {colour[x] for x in adj[u] if colour[x] != -1}
                        if not any(c2 not in taken_u for c2 in lists[u]):
                            valid = False
                            break
                if valid:
                    if backtrack():
                        return True
                colour[best_v] = -1

            return False

        # Order vertices by degree descending for initial ordering hint
        # MRV will handle the rest dynamically
        if backtrack():
            return colour
        return None

    deadline = time.time() + 110.0  # Safety margin before 120s limit

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        result = solve_instance(n, edges, lists, deadline)
        if result is None:
            # Fallback: return a default (shouldn't happen per Alon-Tarsi)
            result = [lists[v][0] for v in range(n)]
        out.append(result)

    # Ensure all values are Python int and lists are list
    final_out = []
    for coloring in out:
        final_out.append([int(c) for c in coloring])

    return final_out
# EVOLVE-BLOCK-END
