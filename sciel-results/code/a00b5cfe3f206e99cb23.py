"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
from random import Random


def construct():
    sys.setrecursionlimit(100000)
    start = time.time()
    TOTAL_BUDGET = 140.0  # explicit guard, safety margin below the 150s kill

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

    def _solve(n, edges, lists, time_limit):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        degree = [len(adj[i]) for i in range(n)]

        # Initialize domains as lists of available colors
        domains = [list(l) for l in lists]
        color = [-1] * n
        uncolored = set(range(n))
        deadline = time.time() + time_limit
        state = {'count': 0, 'ok': False}

        def forward_check(v, c, domains, uncolored):
            """Assign color c to v, propagate constraints to uncolored neighbors.
            Returns (success, trail) where trail is list of (neighbor, removed_color) for undo."""
            trail = []
            for u in adj[v]:
                if u in uncolored:
                    if c in domains[u]:
                        domains[u].remove(c)
                        trail.append((u, c))
                        if not domains[u]:
                            return False, trail
            return True, trail

        def undo(trail, domains):
            for u, c in trail:
                domains[u].append(c)

        def backtrack():
            if not uncolored:
                state['ok'] = True
                return True

            state['count'] += 1
            if (state['count'] & 255) == 0 and time.time() > deadline:
                return False

            # MRV: pick uncolored vertex with fewest domain options
            best_v = -1
            best_len = 999
            best_deg = -1
            for v in uncolored:
                ln = len(domains[v])
                if ln == 0:
                    return False
                if ln < best_len or (ln == best_len and degree[v] > best_deg):
                    best_len = ln
                    best_v = v
                    best_deg = degree[v]
                    if ln == 1:
                        break

            uncolored.discard(best_v)
            color[best_v] = -1  # will be set below

            for c in domains[best_v][:]:  # copy for safety
                color[best_v] = c
                success, trail = forward_check(best_v, c, domains, uncolored)
                if success:
                    if backtrack():
                        return True
                    undo(trail, domains)
                color[best_v] = -1

            uncolored.add(best_v)
            return False

        if backtrack() and state['ok']:
            return [int(c) for c in color]
        return None

    def _greedy(n, adj, lists):
        order = sorted(range(n), key=lambda v: (-len(adj[v]), v))
        color = [-1] * n
        for v in order:
            taken = set()
            for u in adj[v]:
                if color[u] != -1:
                    taken.add(color[u])
            c = -1
            for cc in lists[v]:
                if cc not in taken:
                    c = cc
                    break
            if c == -1:
                for cc in lists[v]:
                    c = cc
                    break
            color[v] = c
        return [int(c) for c in color]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        remaining = TOTAL_BUDGET - (time.time() - start)
        n, edges, lists = instance(splits, seed)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        if remaining > 2.0:
            tl = max(0.2, min(30.0, remaining * 0.7))
            res = _solve(n, edges, lists, tl)
            if res is not None:
                out.append(res)
                continue

        # Fallback: best-effort greedy (may score 0 for a broken instance, but is structurally valid).
        out.append(_greedy(n, adj, lists))

    # Hardening: ensure top-level is a list of 4 lists of Python ints with correct lengths.
    # (A wrong length/type makes the whole result invalid, so guard it.)
    lengths = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, _, _ = instance(splits, seed)
        lengths.append(n)
    for i in range(len(out)):
        if not isinstance(out[i], list):
            out[i] = list(out[i])
        if len(out[i]) != lengths[i]:
            out[i] = (out[i] + [0] * lengths[i])[:lengths[i]]
        out[i] = [int(x) for x in out[i]]
    return out
# EVOLVE-BLOCK-END
