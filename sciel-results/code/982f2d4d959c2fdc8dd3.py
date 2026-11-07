"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
sys.setrecursionlimit(100000)

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

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def solve_csp(n, adj, lists, deadline):
        # MRV + forward checking backtracking
        # domains[v] = set of available colours
        domains = [set(l) for l in lists]
        uncoloured = set(range(n))
        colour = [None] * n

        def mrv_vertex():
            best = None
            best_size = 100
            for v in uncoloured:
                s = len(domains[v])
                if s < best_size:
                    best_size = s
                    best = v
                    if s <= 1:
                        break
            return best

        def propagate(v, c):
            # assign v=c, remove c from neighbours' domains
            removed = []
            for u in adj[v]:
                if u in uncoloured and c in domains[u]:
                    domains[u].discard(c)
                    removed.append((u, c))
                    if not domains[u]:
                        return None  # failure
            return removed

        def backtrack():
            if time.time() > deadline:
                return None
            if not uncoloured:
                return True
            v = mrv_vertex()
            if v is None:
                return True
            # try colours in order of least constraining value (fewest removals)
            for c in sorted(domains[v], key=lambda c: sum(1 for u in adj[v] if u in uncoloured and c in domains[u])):
                uncoloured.discard(v)
                colour[v] = c
                removed = propagate(v, c)
                if removed is None:
                    # undo
                    for u, rc in removed:
                        domains[u].add(rc)
                    uncoloured.add(v)
                    colour[v] = None
                    continue
                if backtrack():
                    return True
                # undo
                for u, rc in removed:
                    domains[u].add(rc)
                uncoloured.add(v)
                colour[v] = None
            return False

        if backtrack():
            return colour
        return None

    def greedy(n, adj, lists):
        colour = [None] * n
        for v in sorted(range(n), key=lambda u: -len(adj[u])):
            taken = {colour[u] for u in adj[v] if colour[u] is not None}
            free = [c for c in lists[v] if c not in taken]
            colour[v] = free[0] if free else lists[v][0]
        return colour

    start = time.time()
    deadline = start + 140.0  # 10s safety margin before 150s kill

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        adj = build_adj(n, edges)

        if time.time() > deadline:
            # fallback: greedy for remaining
            out.append(greedy(n, adj, lists))
            continue

        # Try CSP solver
        result = solve_csp(n, adj, lists, deadline)
        if result is not None:
            # validate
            valid = True
            for v in range(n):
                if result[v] not in lists[v]:
                    valid = False
                    break
                for u in adj[v]:
                    if u > v and result[u] == result[v]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                out.append(result)
                continue

        # fallback: greedy
        out.append(greedy(n, adj, lists))

    # Ensure all are Python ints in lists
    return [[int(c) for c in lst] for lst in out]

# EVOLVE-BLOCK-END
