"""Suite task solver: full backtracking + MRV + forward checking for list colouring."""
# EVOLVE-BLOCK-START
import time

_DEADLINE = 150.0
_TIME_MARGIN = 5.0


def construct():
    t0 = time.time()

    def _deadline():
        return time.time() > (_DEADLINE - _TIME_MARGIN)

    def instance(splits, seed, palette=5):
        from random import Random
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
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        colour = [None] * n
        domains = [list(l) for l in lists]  # each size 3

        def backtrack(assigned_count):
            if _deadline():
                return False
            if assigned_count == n:
                return True
            best_v = -1
            best_d = 10 ** 9
            for v in range(n):
                if colour[v] is None:
                    d = len(domains[v])
                    if d < best_d:
                        best_d = d
                        best_v = v
                        if d == 1:
                            break
            if best_v == -1:
                return True
            if best_d == 0:
                return False
            v = best_v
            for c in domains[v]:
                colour[v] = c
                pruned = []
                ok = True
                for u in adj[v]:
                    if colour[u] is None and c in domains[u]:
                        domains[u].remove(c)
                        pruned.append(u)
                        if not domains[u]:
                            ok = False
                            break
                if ok and backtrack(assigned_count + 1):
                    return True
                for u in pruned:
                    domains[u].append(c)
                colour[v] = None
            return False

        ok = backtrack(0)
        if not ok:
            return None
        return [int(colour[v]) for v in range(n)]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if _deadline():
            break
        n, edges, lists = instance(splits, seed)
        colour = solve(n, edges, lists)
        if colour is None:
            colour = [int(lists[v][0]) for v in range(n)]
        out.append(colour)

    return out
# EVOLVE-BLOCK-END
