"""Clique partition of split graphs: bitmask greedy maximal-clique extraction + bounded merge local search."""
import time
import random
from collections import defaultdict

_DEADLINE = 145.0


def construct():
    t0 = time.time()

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        m = len(edges)
        if m == 0:
            return []

        adj = [0] * n
        for (u, v) in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

        # ---- Structural forced-merge: independent vertices with identical
        # neighbor sets must co-occur in any clique covering their shared edges.
        # We use this to pre-merge cliques later; for now do greedy extraction.
        #
        # Greedy: repeatedly pick an uncovered edge, grow a maximal clique around it.
        rem = set()
        for (u, v) in edges:
            rem.add((u, v))

        deg = defaultdict(int)
        for (u, v) in rem:
            deg[u] += 1
            deg[v] += 1

        result = []
        while rem and time.time() < _DEADLINE:
            # choose edge with min endpoint-degree sum
            (a, b) = min(rem, key=lambda e: deg[e[0]] + deg[e[1]])
            # grow maximal clique containing a and b
            clique = {a, b}
            # candidates adjacent to all current members
            cand = set()
            for v in range(n):
                if v in clique:
                    continue
                ok = True
                for w in clique:
                    if not ((adj[v] >> w) & 1):
                        ok = False
                        break
                if ok:
                    cand.add(v)
            changed = True
            while changed:
                changed = False
                for v in list(cand):
                    ok = True
                    for w in clique:
                        if v == w:
                            continue
                        if not ((adj[v] >> w) & 1):
                            ok = False
                            break
                    if ok:
                        clique.add(v)
                        cand.discard(v)
                        changed = True
            # also try adding candidates that become adjacent after additions (handled above)
            result.append([int(x) for x in sorted(clique)])
            # remove covered edges
            to_remove = set()
            cl = sorted(clique)
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    to_remove.add((cl[i], cl[j]))
            rem -= to_remove
            for e in to_remove:
                if e in rem:
                    deg[e[0]] -= 1
                    deg[e[1]] -= 1

        # ---- Bounded local search: merge cliques whose union is a clique ----
        def is_clique(cl):
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    if not ((adj[cl[i]] >> cl[j]) & 1):
                        return False
            return True

        if len(result) >= 2:
            improved = True
            while improved and time.time() < _DEADLINE:
                improved = False
                nres = len(result)
                for i in range(nres):
                    if time.time() > _DEADLINE:
                        break
                    for j in range(i + 1, nres):
                        merged = result[i] + [x for x in result[j] if x not in result[i]]
                        if is_clique(merged):
                            result[i] = merged
                            result.pop(j)
                            improved = True
                            break
                    if improved:
                        break

        cleaned = [c for c in result if len(c) >= 2]
        return cleaned

    out = []
    for spec in specs:
        if time.time() > _DEADLINE:
            out.append([])
            continue
        out.append(solve_instance(*spec))

    final = []
    for cliques in out:
        cl = []
        for c in cliques:
            if len(c) >= 2:
                cl.append([int(x) for x in c])
        final.append(cl)
    return final
