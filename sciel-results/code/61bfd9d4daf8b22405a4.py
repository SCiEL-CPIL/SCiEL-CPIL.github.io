"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    from random import Random
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

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

    def solve_instance(n, edges, lists):
        """Backtracking with forward checking and MRV heuristic."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Domain for each vertex: colours from its list
        domains = [list(lists[v]) for v in range(n)]
        colour = [-1] * n

        def backtrack(unassigned):
            if not unassigned:
                return True

            # MRV: pick unassigned vertex with fewest available colours
            best_v = -1
            best_count = 10
            for v in unassigned:
                # Count available colours considering assigned neighbours
                taken = set()
                for u in adj[v]:
                    if colour[u] != -1:
                        taken.add(colour[u])
                avail = [c for c in domains[v] if c not in taken]
                if len(avail) < best_count:
                    best_count = len(avail)
                    best_v = v
                    if best_count == 0:
                        break
                    if best_count == 1:
                        break

            if best_v == -1:
                return False
            if best_count == 0:
                return False

            taken = set()
            for u in adj[best_v]:
                if colour[u] != -1:
                    taken.add(colour[u])
            avail = [c for c in domains[best_v] if c not in taken]

            for c in avail:
                colour[best_v] = c
                new_unassigned = [v for v in unassigned if v != best_v]
                if backtrack(new_unassigned):
                    return True
                colour[best_v] = -1

            return False

        if backtrack(list(range(n))):
            return colour
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # Return best effort: use greedy fallback for remaining
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
            out.append(colour)
            continue

        n, edges, lists = instance(splits, seed)
        result = solve_instance(n, edges, lists)
        if result is None:
            # Fallback: greedy
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = [0] * n
            for v in range(n):
                taken = {colour[u] for u in adj[v]}
                free = [c for c in lists[v] if c not in taken]
                colour[v] = free[0] if free else lists[v][0]
            out.append(colour)
        else:
            out.append(result)

    return out
# EVOLVE-BLOCK-END
