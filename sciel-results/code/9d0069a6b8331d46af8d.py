"""Decompose fixed Erdős-Gallai instance graphs into few cycles and edges."""

def construct():
    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(n, edges):
        # Build adjacency and remaining edge set
        adj = {i: set() for i in range(n)}
        edge_set = set()
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))

        remaining = set(edge_set)
        pieces = []

        def has_edge(a, b):
            e = (a, b) if a < b else (b, a)
            return e in remaining

        def remove_edge(a, b):
            e = (a, b) if a < b else (b, a)
            remaining.discard(e)

        def rem_degree(v):
            return sum(1 for w in adj[v] if has_edge(v, w))

        # Try to find a cycle starting from start using randomized greedy walk
        # with occasional backtracking. Returns list of vertices (cycle) or None.
        def find_cycle(start, max_steps=500):
            # Use randomized DFS with cycle detection
            import random as _rnd
            # Try multiple random starts of the walk
            best_cycle = None
            for attempt in range(5):
                path = [start]
                onpath = {start}
                current = start
                # Walk as long as possible
                for _ in range(max_steps):
                    nbrs = [w for w in adj[current] if has_edge(current, w)]
                    if not nbrs:
                        break
                    # If start is reachable and path length >= 3, we have a cycle
                    if start in nbrs and len(path) >= 3:
                        cyc = path[:]
                        if best_cycle is None or len(cyc) > len(best_cycle):
                            best_cycle = cyc
                        break
                    # Prefer neighbors not on path
                    ext = [w for w in nbrs if w not in onpath]
                    if ext:
                        # Pick one with highest remaining degree
                        ext.sort(key=lambda w: rem_degree(w), reverse=True)
                        # Add some randomness
                        idx = 0
                        if len(ext) > 1 and _rnd.random() < 0.3:
                            idx = _rnd.randint(1, len(ext) - 1)
                        current = ext[idx]
                        path.append(current)
                        onpath.add(current)
                    else:
                        # All neighbors are on path; check if any closes to start
                        if start in nbrs and len(path) >= 3:
                            cyc = path[:]
                            if best_cycle is None or len(cyc) > len(best_cycle):
                                best_cycle = cyc
                            break
                        # Dead end
                        break
                if best_cycle is not None and len(best_cycle) > 3:
                    break  # Good enough
            return best_cycle

        # Better: use a systematic approach to find long cycles
        # First, try to find a Hamiltonian cycle or near-Hamiltonian
        # Using the fact that these are relatively sparse graphs

        def find_long_cycle(start):
            """Find a cycle through start using DFS that tries to maximize length."""
            best = None
            # Iterative DFS
            stack = [(start, [start], {start})]
            visited_nodes = 0
            max_visit = 50000
            while stack:
                if visited_nodes > max_visit:
                    break
                node, path, onpath = stack.pop()
                visited_nodes += 1

                nbrs = [w for w in adj[node] if has_edge(node, w)]
                # Check if we can close a cycle
                if start in nbrs and len(path) >= 3:
                    cyc = path[:]
                    if best is None or len(cyc) > len(best):
                        best = cyc
                        # Don't break; keep looking for longer
                # Extend
                ext = [w for w in nbrs if w not in onpath]
                # Sort by remaining degree descending
                ext.sort(key=lambda w: rem_degree(w), reverse=True)
                for w in ext:
                    stack.append((w, path + [w], onpath | {w}))
            return best

        # Main loop: extract cycles greedily
        while remaining:
            # Find a vertex with degree >= 2
            candidates = [v for v in range(n) if rem_degree(v) >= 2]
            if not candidates:
                break

            # Try to find a cycle from the highest-degree vertex
            candidates.sort(key=rem_degree, reverse=True)
            found = False
            best_cycle = None
            best_len = 0

            for start in candidates[:10]:  # Try top 10 candidates
                cyc = find_long_cycle(start)
                if cyc is not None and len(cyc) > best_len:
                    best_len = len(cyc)
                    best_cycle = cyc
                    if len(cyc) >= n:  # Hamiltonian cycle
                        break

            if best_cycle is None:
                # No cycles found; remaining edges are single-edge pieces
                for (u, v) in sorted(remaining):
                    pieces.append([u, v])
                remaining.clear()
                break

            # Add the cycle and remove its edges
            pieces.append(best_cycle)
            for i in range(len(best_cycle)):
                a = best_cycle[i]
                b = best_cycle[(i + 1) % len(best_cycle)]
                remove_edge(a, b)

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(n, edges)
        results.append(pieces)
    return results
