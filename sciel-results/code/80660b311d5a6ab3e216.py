"""Decompose graphs into edge-disjoint cycles and edges, minimizing piece count."""
import time
import random as _random

def construct():
    deadline = time.time() + 140
    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_all_cycles_greedy(edges, rng, time_limit=3.0):
        """Greedy cycle extraction. Returns (cycles, remaining_edges)."""
        start_t = time.time()
        remaining = set(edges)
        cycles = []

        def build_adj(es):
            a = {}
            for u, v in es:
                a.setdefault(u, set()).add(v)
                a.setdefault(v, set()).add(u)
            return a

        while remaining and time.time() - start_t < time_limit:
            adj = build_adj(remaining)
            deg2 = [u for u in adj if len(adj[u]) >= 2]
            if not deg2:
                break

            best_cycle = None
            # Try several random starts
            for _ in range(30):
                start = rng.choice(deg2)
                # DFS to find a cycle through start
                path = [start]
                used_e = set()

                def dfs(curr):
                    if len(path) > 25:
                        return None
                    for nb in list(adj.get(curr, [])):
                        e = (min(curr, nb), max(curr, nb))
                        if e not in remaining or e in used_e:
                            continue
                        if nb == start and len(path) >= 3:
                            return list(path)
                        path.append(nb)
                        used_e.add(e)
                        res = dfs(nb)
                        if res:
                            return res
                        path.pop()
                        used_e.discard(e)
                    return None

                c = dfs(start)
                if c:
                    # Verify
                    ok = True
                    for i in range(len(c)):
                        u, v = c[i], c[(i + 1) % len(c)]
                        if (min(u, v), max(u, v)) not in remaining:
                            ok = False
                            break
                    if ok and (best_cycle is None or len(c) > len(best_cycle)):
                        best_cycle = c
                        if len(c) >= 10:
                            break

            if best_cycle:
                for i in range(len(best_cycle)):
                    u, v = best_cycle[i], best_cycle[(i + 1) % len(best_cycle)]
                    remaining.discard((min(u, v), max(u, v)))
                cycles.append(best_cycle)
            else:
                break

        return cycles, remaining

    def try_improve(pieces, edges, rng, time_limit=3.0):
        """Try to find cycles among single-edge pieces."""
        start_t = time.time()
        singles = [tuple(p) for p in pieces if len(p) == 2]
        cycles = [p for p in pieces if len(p) >= 3]

        if not singles:
            return pieces

        adj = {}
        for u, v in singles:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        # Repeatedly try to find a cycle in the single-edge subgraph
        for _ in range(100):
            if time.time() - start_t > time_limit:
                break
            deg2 = [u for u in adj if len(adj[u]) >= 2]
            if not deg2:
                break
            start = rng.choice(deg2)
            path = [start]
            used = set()

            def dfs(curr):
                if len(path) > 15:
                    return None
                for nb in list(adj.get(curr, [])):
                    e = (min(curr, nb), max(curr, nb))
                    if e in used:
                        continue
                    if nb == start and len(path) >= 3:
                        return list(path)
                    path.append(nb)
                    used.add(e)
                    res = dfs(nb)
                    if res:
                        return res
                    path.pop()
                    used.discard(e)
                return None

            c = dfs(start)
            if c:
                ce = set()
                for i in range(len(c)):
                    u, v = c[i], c[(i + 1) % len(c)]
                    ce.add((min(u, v), max(u, v)))
                new_singles = [e for e in singles if e not in ce]
                # Update adj
                for e in ce:
                    u, v = e
                    if v in adj.get(u, set()):
                        adj[u].discard(v)
                    if u in adj.get(v, set()):
                        adj[v].discard(u)
                singles = new_singles
                cycles.append(c)
                if not singles:
                    break

        return cycles + [list(e) for e in singles]

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))

    for idx, (n, p, seed) in enumerate(specs):
        if time.time() > deadline - 15:
            break
        edges = instance(n, p, seed)
        rng = Random(seed * 7919 + idx * 31)

        # Greedy cycle extraction
        cycles, remaining = find_all_cycles_greedy(edges, rng, time_limit=2.5)
        pieces = cycles + [list(e) for e in remaining]

        # Try to improve
        if time.time() < deadline - 10:
            pieces = try_improve(pieces, edges, rng, time_limit=2.5)

        # Validate
        ec = {}
        for p in pieces:
            if len(p) == 2:
                e = tuple(sorted(p))
                ec[e] = ec.get(e, 0) + 1
            else:
                for i in range(len(p)):
                    u, v = p[i], p[(i + 1) % len(p)]
                    e = tuple(sorted((u, v)))
                    ec[e] = ec.get(e, 0) + 1

        valid = all(ec.get(e, 0) == 1 for e in edges) and len(ec) == len(edges)
        if not valid:
            pieces = [list(e) for e in edges]

        results.append(pieces)

    while len(results) < 3:
        idx = len(results)
        n, p, seed = specs[idx]
        edges = instance(n, p, seed)
        results.append([list(e) for e in edges])

    return results
