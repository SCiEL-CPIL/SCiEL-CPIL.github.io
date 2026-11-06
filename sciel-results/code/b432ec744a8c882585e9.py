import time
import random
from collections import deque

_START = time.time()
DEADLINE = _START + 140.0


def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose(edges, n):
    """Decompose edge set into edge-disjoint cycles + leftover single edges,
    minimizing piece count. A cycle of length L replaces L single-edge pieces
    with 1, so longer cycles are strictly better."""
    rem = {i: set() for i in range(n)}
    for u, v in edges:
        rem[u].add(v)
        rem[v].add(u)

    pieces = []

    def _deg(u):
        return len(rem[u])

    def _greedy_long_cycle(start, max_iter=2000):
        """Greedy randomized DFS to find a long cycle from start.
        Returns list of vertices in cycle order, or None."""
        if not rem[start]:
            return None
        best = None
        for _ in range(max_iter):
            if time.time() > DEADLINE:
                break
            path = [start]
            visited = {start}
            cur = start
            while True:
                # neighbors in remaining graph
                nbrs = list(rem[cur])
                if not nbrs:
                    break
                unvis = [v for v in nbrs if v not in visited]
                if unvis:
                    # pick neighbor with highest remaining degree to keep options open
                    nxt = max(unvis, key=lambda x: _deg(x))
                    path.append(nxt)
                    visited.add(nxt)
                    cur = nxt
                else:
                    # all neighbors visited; try to close cycle to start
                    if start in rem[cur]:
                        cyc = path[:]
                        if best is None or len(cyc) > len(best):
                            best = cyc
                    break
        return best

    def _dfs_any_cycle(start):
        """Find any cycle starting from start using DFS. Returns cycle or None."""
        parent = {start: -1}
        visited = {start}
        stack = [start]
        while stack:
            u = stack[-1]
            # find an unprocessed neighbor
            found_next = None
            for v in list(rem[u]):
                if v == parent.get(u, -1):
                    continue
                if v in visited:
                    # reconstruct cycle
                    if v == start:
                        cyc = [u]
                        cur = u
                        while cur != v:
                            cur = parent[cur]
                            cyc.append(cur)
                        cyc.reverse()
                        if len(cyc) >= 3:
                            return cyc
                        continue
                    # back edge
                    cyc = [u]
                    cur = u
                    while cur != v:
                        cur = parent[cur]
                        cyc.append(cur)
                    cyc.reverse()
                    if len(cyc) >= 3:
                        return cyc
                    continue
                found_next = v
                break
            if found_next is not None:
                parent[found_next] = u
                visited.add(found_next)
                stack.append(found_next)
            else:
                stack.pop()
        return None

    def _find_any_cycle():
        """Find any cycle in the remaining graph."""
        for s in range(n):
            if rem[s]:
                cyc = _dfs_any_cycle(s)
                if cyc is not None:
                    return cyc
        return None

    # Main loop: repeatedly find cycles and remove them
    # Strategy: try to find longer cycles by starting from high-degree vertices
    # and using a DFS that explores deeper paths

    while time.time() < DEADLINE:
        # Get vertices with remaining edges, sorted by degree
        degs = [(_deg(i), i) for i in range(n) if rem[i]]
        if not degs:
            break
        degs.sort(reverse=True)

        found = None
        best_found_len = 0

        # Try several starting vertices, prefer high-degree ones
        for _, start in degs[:10]:
            if time.time() > DEADLINE:
                break
            # Try greedy long cycle
            cyc = _greedy_long_cycle(start)
            if cyc is not None and len(cyc) > best_found_len:
                best_found_len = len(cyc)
                found = cyc
            # Early exit if we found a very long cycle
            if best_found_len > 14:
                break

        if found is None:
            # Fall back to finding any cycle
            found = _find_any_cycle()

        if found is None:
            break

        # Remove cycle edges from rem
        cyc = found
        for i in range(len(cyc)):
            a, b = cyc[i], cyc[(i + 1) % len(cyc)]
            rem[a].discard(b)
            rem[b].discard(a)
        pieces.append(cyc)

    # Leftover edges
    leftover = set()
    for u in range(n):
        for v in rem[u]:
            if u < v:
                leftover.add((u, v))
    for u, v in sorted(leftover):
        pieces.append([u, v])

    return pieces


def construct():
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = _instance(n, p, seed)
        pieces = _decompose(edges, n)
        results.append(pieces)
    return results
