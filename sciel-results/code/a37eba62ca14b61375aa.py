import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # margin for 120s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def norm(e):
        return (min(e[0], e[1]), max(e[0], e[1]))

    def find_longest_cycle(adj, n, remaining_edges, deadline):
        """Find a long simple cycle using DFS with backtracking, trying each start."""
        best_len = 0
        best_cycle = []

        # Order starts by degree descending to find long cycles faster
        starts = sorted(range(n), key=lambda x: -len(adj[x]))

        for start in starts:
            if len(adj[start]) < 1:
                continue
            if time.time() > deadline:
                break

            # DFS with explicit stack to find cycles through `start`
            # We do a bounded DFS: build a path from start, and if we hit start again with len>=3, record.
            # Use recursion with depth limit and branch pruning.
            best_local = [0, []]

            # We'll do a DFS that explores paths starting at start; when we can return to start, record cycle.
            path = [start]
            in_path = {start}

            def dfs(u, depth):
                nonlocal best_local
                if time.time() > deadline:
                    return
                # Try to close to start
                if depth >= 2 and start in adj[u] and norm((u, start)) in remaining_edges:
                    cyc = path + [start]
                    # cycle vertices path (start..u); length = len(path)
                    if len(path) > best_local[0]:
                        best_local[0] = len(path)
                        best_local[1] = list(path)
                for v in sorted(adj[u]):
                    if norm((u, v)) not in remaining_edges:
                        continue
                    if v == start:
                        # handled above (closing)
                        continue
                    if v in in_path:
                        continue
                    # prune: even taking all remaining reachable, can we beat best?
                    path.append(v)
                    in_path.add(v)
                    dfs(v, depth + 1)
                    in_path.remove(v)
                    path.pop()

            dfs(start, 0)
            if best_local[0] > best_len:
                best_len = best_local[0]
                best_cycle = best_local[1]

        return best_cycle

    def decompose_one(spec, deadline):
        edges = instance(*spec)
        n = spec[0]
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        remaining_edges = set(edges)
        pieces = []

        # Phase 1: greedily extract long cycles
        while remaining_edges and time.time() < deadline:
            best_cycle = find_longest_cycle(adj, n, remaining_edges, deadline)
            if not best_cycle or len(best_cycle) < 3:
                break
            pieces.append(best_cycle)
            L = len(best_cycle)
            for i in range(L):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % L]
                e = norm((u, v))
                if e in remaining_edges:
                    remaining_edges.discard(e)
                    adj[u].discard(v)
                    adj[v].discard(u)

        # Phase 2: local improvement — try to form cycles from leftover edges + single edges
        # Represent leftover as a multigraph; try to find any cycle among leftover edges.
        # (find_longest_cycle already handles this; loop above already exhausted cycles.)

        # Phase 3: remaining edges become single-edge pieces
        for e in remaining_edges:
            pieces.append(list(e))

        return pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    per_spec_budget = deadline - start_time

    for i, spec in enumerate(specs):
        # Allocate remaining time across remaining specs
        remaining_specs = len(specs) - i
        spec_deadline = min(deadline, start_time + (deadline - start_time) * remaining_specs / max(1, remaining_specs))
        # ensure at least a little time
        pieces = decompose_one(spec, spec_deadline)
        results.append(pieces)

    return results
