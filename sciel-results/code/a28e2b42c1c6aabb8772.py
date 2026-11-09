# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        # Build adjacency as dict vertex -> set of neighbors (remaining edges)
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        # Degree for tie-breaking
        degree = {v: len(s) for v, s in adj.items()}

        pieces = []
        used = set()
        start_time = time.time()
        deadline = start_time + 110.0  # safety margin under 120s

        # Find a long cycle via DFS, preferring high-degree neighbors to keep graph dense
        def find_cycle():
            # Try starting from a few candidate vertices
            # Prefer vertices with high degree
            candidates = sorted(adj.keys(), key=lambda x: (-degree[x], x))
            for start in candidates:
                if not adj.get(start):
                    continue
                # DFS with path tracking
                best_cycle = None
                # Iterative DFS
                # We want a cycle: start -> ... -> start
                # Use recursion limit safe iterative approach
                # Track visited in current path
                path = [start]
                on_path = {start}
                parent = {start: None}
                stack = [start]
                # We'll do a DFS and when we hit start again via a neighbor, we found a cycle
                # But standard DFS: when we see a neighbor already in on_path (and not the immediate parent), cycle found
                # To get a LONG cycle, we prefer to go deep.
                # Use a recursive-like DFS with explicit stack storing (node, iterator)
                # Simpler: do a DFS that explores in order of high degree, and the first cycle found
                # that closes to start is used. But that may be short.
                # Better: find ANY cycle, but bias to long by exploring high-degree first.
                # Let's do: DFS from start, when we encounter a neighbor w that is in on_path
                # and w == start, we have a cycle path[start:] + [start].
                # Also if w is in on_path but w != start, we have a cycle but not necessarily through start.
                # For simplicity, we only accept cycles that close to start.
                # To get long cycles, start from high-degree and explore deeply.
                # Implement iterative DFS:
                stack = [(start, iter(sorted(adj[start], key=lambda x: (-degree[x], x))))]
                path = [start]
                on_path = {start}
                while stack:
                    node, it = stack[-1]
                    found = False
                    for nb in it:
                        if nb not in adj or not adj[nb]:
                            continue
                        if nb == start and len(path) >= 3:
                            # Cycle found: path + [start]
                            cycle = list(path)
                            return cycle
                        if nb in on_path:
                            # Cycle not through start; skip for now (we want cycles through start)
                            continue
                        # Explore nb
                        on_path.add(nb)
                        path.append(nb)
                        stack.append((nb, iter(sorted(adj[nb], key=lambda x: (-degree[x], x)))))
                        found = True
                        break
                    if not found:
                        # Backtrack
                        stack.pop()
                        path.pop()
                        on_path.discard(node)
            return None

        while True:
            if time.time() > deadline:
                break
            # Remove isolated vertices
            changed = True
            while changed:
                changed = False
                for v in list(adj.keys()):
                    if not adj[v]:
                        del adj[v]
                        changed = True
            if not adj:
                break
            cycle = find_cycle()
            if cycle is None:
                break
            # Remove cycle edges from adjacency
            for i in range(len(cycle)):
                a = cycle[i]
                b = cycle[(i + 1) % len(cycle)]
                adj[a].discard(b)
                adj[b].discard(a)
                degree[a] = len(adj[a])
                degree[b] = len(adj[b])
            pieces.append(cycle)

        # Remaining edges as single-edge pieces
        for u, v in edges:
            # Check if this edge is still present (i.e., not part of any cycle)
            # We track used edges via adjacency: if v in adj.get(u, set()), it's remaining
            if v in adj.get(u, set()):
                pieces.append([u, v])
                adj[u].discard(v)
                adj[v].discard(u)

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        results.append(decompose(edges))
    return results
# EVOLVE-BLOCK-END
