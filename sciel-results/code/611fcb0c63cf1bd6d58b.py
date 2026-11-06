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
    minimizing piece count."""
    rem = {i: set() for i in range(n)}
    for u, v in edges:
        rem[u].add(v)
        rem[v].add(u)

    pieces = []

    def find_cycle_dfs(start):
        """Find a cycle using DFS from start. Returns cycle as list of vertices, or None."""
        best_cycle = None
        best_len = 0

        # Iterative DFS with explicit stack
        # stack entries: (node, neighbor_iter)
        parent = {start: -1}
        on_stack = {start}
        stack = [(start, iter(sorted(rem[start], key=lambda x: -len(rem[x]))))]

        while stack:
            if time.time() > DEADLINE:
                return best_cycle
            node, neighbors = stack[-1]
            advanced = False
            for nb in neighbors:
                if nb == parent.get(node, -1):
                    continue
                if nb in on_stack:
                    # Found a cycle: extract from nb to node via parent chain
                    cyc = [node]
                    cur = node
                    while cur != nb:
                        cur = parent[cur]
                        cyc.append(cur)
                    cyc.reverse()
                    if len(cyc) > best_len:
                        best_len = len(cyc)
                        best_cycle = cyc[:]
                    # Try to find longer cycles by continuing
                    # But for efficiency, return if we have a decent cycle
                    if best_len >= 10:
                        return best_cycle
                else:
                    parent[nb] = node
                    on_stack.add(nb)
                    stack.append((nb, iter(sorted(rem[nb], key=lambda x: -len(rem[x])))))
                    advanced = True
                    break
            if not advanced:
                stack.pop()
                on_stack.discard(node)
        return best_cycle

    def find_cycle_bfs():
        """Find any cycle in the remaining graph using BFS."""
        for start in range(n):
            if not rem[start]:
                continue
            parent = {start: -1}
            visited = {start}
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in list(rem[u]):
                    if v == parent.get(u, -1):
                        continue
                    if v in visited:
                        # Found a cycle
                        cyc = [u]
                        cur = u
                        while cur != v:
                            cur = parent[cur]
                            cyc.append(cur)
                        cyc.reverse()
                        if len(cyc) >= 3:
                            return cyc
                    else:
                        parent[v] = u
                        visited.add(v)
                        queue.append(v)
        return None

    # Main loop: repeatedly find cycles and remove them
    while time.time() < DEADLINE:
        # Get vertices with remaining edges, sorted by degree
        degs = [(len(rem[i]), i) for i in range(n) if rem[i]]
        if not degs:
            break
        degs.sort(reverse=True)

        found = None
        best_found_len = 0

        # Try several starting vertices, prefer high-degree ones
        for _, start in degs[:10]:
            if time.time() > DEADLINE:
                break
            cyc = find_cycle_dfs(start)
            if cyc is not None and len(cyc) > best_found_len:
                best_found_len = len(cyc)
                found = cyc
            # Early exit if we found a very long cycle
            if best_found_len > 15:
                break

        if found is None:
            # Fall back to finding any cycle
            found = find_cycle_bfs()

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
        if time.time() > DEADLINE:
            edges = _instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        edges = _instance(n, p, seed)
        pieces = _decompose(edges, n)
        results.append(pieces)
    return results
