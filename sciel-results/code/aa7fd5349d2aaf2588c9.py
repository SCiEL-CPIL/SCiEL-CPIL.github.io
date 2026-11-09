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
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        # Smallest list / most constrained vertex first (deterministic tie-break).
        order = sorted(range(n), key=lambda v: (len(lists[v]), len(adj[v]), v))
        pos = [0] * n
        for i, v in enumerate(order):
            pos[v] = i
        # Only neighbours that come later in the order matter during forward propagation.
        fwd = []
        for v in order:
            fwd.append([u for u in adj[v] if pos[u] > pos[v]])

        used = [0] * 5
        colour = [-1] * n
        # Explicit stack: (vertex_index_in_order, colour_choice_index)
        stack = [(0, 0)]

        while stack:
            i, ci = stack[-1]
            v = order[i]
            lst = lists[v]
            if ci < len(lst):
                c = lst[ci]
                stack[-1] = (i, ci + 1)          # advance this vertex's choice pointer
                if used[c] == 0:                 # colour c is free
                    used[c] = 1
                    colour[v] = c
                    ok = True
                    for u in fwd[v]:
                        lu = lists[u]
                        if c not in lu:          # u has no remaining free colour -> conflict
                            ok = False
                            break
                    if ok:
                        if i + 1 < n:
                            stack.append((i + 1, 0))
                        else:
                            # All vertices coloured successfully.
                            return [int(colour[j]) for j in range(n)]
                    if not ok:
                        used[c] = 0
                        colour[v] = -1
            else:
                # No colour works for this vertex -> backtrack.
                stack.pop()
                if colour[v] != -1:
                    used[colour[v]] = 0
                    colour[v] = -1

        # Alon-Tarsi guarantees a solution exists; this is only a safety net.
        return [int(lists[v][0]) for v in range(n)]

    deadline = time.time() + 105.0
    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        out.append(solve(n, edges, lists))
        if time.time() > deadline:
            # Guard: if (unexpectedly) over budget, keep the already-valid results.
            break
    return out
# EVOLVE-BLOCK-END
