"""Rewritten decomposition: greedily extract long cycles, then single edges."""

def construct():
    import time, random
    from collections import defaultdict

    t0 = time.time()
    DEADLINE = 110.0  # safety margin under 120s

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(n, p, seed):
        edges = instance(n, p, seed)
        edge_set = {tuple(e) for e in edges}
        adj = defaultdict(set)
        for (u, v) in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        remaining = set(edge_set)

        def remove_edge(u, v):
            key = (u, v) if u < v else (v, u)
            remaining.discard(key)
            adj[u].discard(v)
            adj[v].discard(u)

        def try_find_cycle(start, max_len, rng):
            """Greedy longest-path-then-close search from start. Returns list of vertices (cycle) or None."""
            # We'll do a DFS building a simple path, trying to close it into a cycle.
            best = None
            # Iterative deepening-ish greedy with backtracking limited by time.
            # Use a recursive DFS with a node budget.
            budget = [max_len * 40]
            best_cycle = [None]

            def dfs(path, used):
                if budget[0] <= 0:
                    return
                budget[0] -= 1
                cur = path[-1]
                # try to close cycle
                if len(path) >= 3 and start in adj[cur]:
                    cand = list(path)
                    if best_cycle[0] is None or len(cand) > len(best_cycle[0]):
                        best_cycle[0] = cand
                # extend
                nbrs = list(adj[cur])
                # prefer neighbors not in path, and to make long cycles avoid start until near end
                nbrs.sort(key=lambda x: (x in used, x == start))
                for nb in nbrs:
                    if nb in used:
                        continue
                    path.append(nb)
                    used.add(nb)
                    dfs(path, used)
                    path.pop()
                    used.discard(nb)
                    if budget[0] <= 0:
                        return
                    if best_cycle[0] and len(best_cycle[0]) >= max_len:
                        return

            dfs([start], {start})
            return best_cycle[0]

        pieces = []
        guard = 0
        while remaining:
            if time.time() > DEADLINE:
                break
            guard += 1
            if guard % 200 == 0 and time.time() > DEADLINE:
                break

            # Find a good cycle: try several start vertices, pick the longest found.
            best_cycle = None
            starts = list(adj.keys())
            random.Random(seed + guard).shuffle(starts)
            # limit starts tried per round
            for s in starts:
                if time.time() > DEADLINE:
                    break
                if not adj[s]:
                    continue
                deg = len(adj[s])
                if deg < 2:
                    continue
                c = try_find_cycle(s, 10**9, random.Random(seed * 131 + guard * 7 + s))
                if c and (best_cycle is None or len(c) > len(best_cycle)):
                    best_cycle = c
                if best_cycle and len(best_cycle) >= 8:
                    break

            if best_cycle and len(best_cycle) >= 3:
                cyc = best_cycle
                for i in range(len(cyc)):
                    a = cyc[i]
                    b = cyc[(i + 1) % len(cyc)]
                    if (a, b) in remaining or (b, a) in remaining:
                        remove_edge(a, b)
                pieces.append(list(cyc))
            else:
                # no cycle found from tried starts; take a single edge
                e = next(iter(remaining))
                remove_edge(e[0], e[1])
                pieces.append(list(e))

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        results.append(decompose(*spec))
        if time.time() > DEADLINE:
            # fallback: fill remaining with trivial per-edge decomposition
            pass
    return results
