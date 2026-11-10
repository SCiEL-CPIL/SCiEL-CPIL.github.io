"""Model-and-optimize: parameterise the colouring (one int per vertex) and fit
each instance directly against the objective (list-respecting, edge-different
colouring).  Greedy + local repair, swept over several vertex orderings."""
# EVOLVE-BLOCK-START
import time
from random import Random


def construct():
    t0 = time.time()
    LIMIT = 120.0          # hard wall-clock cap for the evaluator
    MARGIN = 18.0          # safety margin before the cap (return early)

    def deadline():
        return (time.time() - t0) > (LIMIT - MARGIN)

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

    def try_color(n, adj, lists, order, palette=5):
        colour = [None] * n
        for v in order:
            taken = set()
            for nb in adj[v]:
                c = colour[nb]
                if c is not None:
                    taken.add(c)
            free_c = None
            for c in lists[v]:
                if c not in taken:
                    free_c = c
                    break
            if free_c is not None:
                colour[v] = free_c
            else:
                # list constraint violated -> pick least-used list colour
                counts = [0, 0, 0, 0, 0]
                for nb in adj[v]:
                    c = colour[nb]
                    if c is not None:
                        counts[c] += 1
                colour[v] = min(lists[v], key=lambda c: (counts[c], c))

        # Repair / local optimisation against the objective.
        changed = True
        guard = 0
        while changed:
            changed = False
            guard += 1
            if guard > 60 * n:            # hard safety bound (should not trigger)
                break
            if deadline():
                break
            for v in order:
                for nb in adj[v]:
                    if colour[nb] == colour[v]:
                        counts = [0, 0, 0, 0, 0]
                        for x in adj[v]:
                            counts[colour[x]] += 1
                        free = [c for c in lists[v]
                                if c != colour[v] and counts[c] == 0]
                        if free:
                            colour[v] = free[0]
                        else:
                            candidates = [c for c in lists[v] if c != colour[v]]
                            if candidates:
                                colour[v] = min(candidates, key=lambda c: (counts[c], c))
                        changed = True
                        break
        return colour

    def valid(colour, lists, edges):
        for i, c in enumerate(colour):
            if c not in lists[i]:
                return False
        for u, v in edges:
            if colour[u] == colour[v]:
                return False
        return True

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if deadline():
            break
        n, edges, lists = instance(splits, seed)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        deg = [len(adj[v]) for v in range(n)]

        # Degeneracy ordering (smallest remaining degree first) + its reverse.
        rem = deg[:]
        used = [False] * n
        degen = []
        for _ in range(n):
            m = 0
            mi = 0
            for i in range(n):
                if not used[i] and rem[i] < m:
                    m = rem[i]
                    mi = i
            degen.append(mi)
            used[mi] = True
            for nb in adj[mi]:
                if not used[nb]:
                    rem[nb] -= 1

        r = Random(9973 + seed)
        orders = [
            sorted(range(n), key=lambda v: (-deg[v], v)),                 # high degree first
            sorted(range(n), key=lambda v: (deg[v], v)),                  # low degree first
            degen,                                                        # degeneracy order
            list(reversed(degen)),                                        # reverse degeneracy
            list(range(n)),                                               # natural order
            list(r.sample(range(n), n)),                                  # random shuffle 1
            list(r.sample(range(n), n)),                                  # random shuffle 2
            list(r.sample(range(n), n)),                                  # random shuffle 3
        ]

        best = None
        for order in orders:
            col = try_color(n, adj, lists, order)
            if valid(col, lists, edges):
                best = col
                break
            if best is None:
                best = col
            if deadline():
                break
        out.append([int(c) for c in best])

    return out
# EVOLVE-BLOCK-END
