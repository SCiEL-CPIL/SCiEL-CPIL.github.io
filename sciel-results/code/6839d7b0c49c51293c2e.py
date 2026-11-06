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

    def find_cycle_dfs(start, max_len=None):
        """Find a cycle using DFS from start. Returns cycle as list of vertices, or None."""
        best_cycle = None
        best_len = 0

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
                    cyc = [node]
                    cur = node
                    while cur != nb:
                        cur = parent[cur]
                        cyc.append(cur)
                    cyc.reverse()
                    if len(cyc) > best_len:
                        best_len = len(cyc)
                        best_cycle = cyc[:]
                    if max_len and best_len >= max_len:
                        return best_cycle
                    if best_len >= 12:
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

    def find_longest_cycle_2opt(start, initial_cycle):
        """Try to improve cycle length using 2-opt style swaps."""
        if not initial_cycle:
            return None
        cyc = list(initial_cycle)
        improved = True
        best_cyc = cyc[:]
        best_len = len(cyc)

        while improved and time.time() < DEADLINE:
            improved = False
            L = len(cyc)
            for i in range(L):
                if time.time() > DEADLINE:
                    break
                a = cyc[i]
                b = cyc[(i + 1) % L]
                for j in range(i + 2, L):
                    if j == (i + 1) % L:
                        continue
                    c = cyc[j]
                    d = cyc[(j + 1) % L]
                    # Try reversing segment i+1..j
                    # New cycle: a, d, c, ..., b
                    # Check if edge (a,d) and (b,c) exist
                    if d in rem[a] and c in rem[b]:
                        # Verify the reversed segment forms valid edges
                        new_cyc = [a]
                        # d, then reverse from c back to b's position
                        seg = cyc[(j+1)%L:i+1 if i+1 <= j else L]
                        # Actually let me think more carefully
                        # Original: a, cyc[i+1], ..., cyc[j], d, ...
                        # New: a, d, cyc[j], cyc[j-1], ..., cyc[i+1], ...
                        # We need edge (a,d) and edge (cyc[i+1], c) where c=cyc[j]
                        # Wait, the segment from i+1 to j gets reversed
                        # So new sequence: a, d, cyc[j], cyc[j-1], ..., cyc[i+1], then continue from after j
                        # We need: edge(a,d) exists, and edge(cyc[i+1], cyc[j]) exists
                        # cyc[i+1] is b, cyc[j] is c
                        # So we need edge(a,d) and edge(b,c)
                        # b = cyc[i+1], c = cyc[j]
                        if b in rem[c] and a in rem[d]:
                            new_cyc = [a, d]
                            # Add reversed segment from j down to i+1
                            for k in range(j, i, -1):
                                new_cyc.append(cyc[k])
                            # Add the rest: from j+1 to end, then start to i
                            for k in range(j + 1, L):
                                new_cyc.append(cyc[k])
                            for k in range(0, i):
                                new_cyc.append(cyc[k])
                            # Verify all edges in new_cyc
                            valid = True
                            for idx in range(len(new_cyc)):
                                u2 = new_cyc[idx]
                                v2 = new_cyc[(idx + 1) % len(new_cyc)]
                                if v2 not in rem[u2]:
                                    valid = False
                                    break
                            if valid and len(new_cyc) > best_len:
                                best_cyc = new_cyc
                                best_len = len(new_cyc)
                                cyc = new_cyc
                                improved = True
                                break
                if improved:
                    break
        return best_cyc

    # Main loop: repeatedly find cycles and remove them
    while time.time() < DEADLINE:
        degs = [(len(rem[i]), i) for i in range(n) if rem[i]]
        if not degs:
            break
        degs.sort(reverse=True)

        found = None
        best_found_len = 0

        # Try several starting vertices, prefer high-degree ones
        for _, start in degs[:15]:
            if time.time() > DEADLINE:
                break
            cyc = find_cycle_dfs(start)
            if cyc is not None:
                # Try to improve with 2-opt
                improved_cyc = find_longest_cycle_2opt(start, cyc)
                if improved_cyc is not None and len(improved_cyc) > len(cyc):
                    cyc = improved_cyc
                if len(cyc) > best_found_len:
                    best_found_len = len(cyc)
                    found = cyc
            if best_found_len > 18:
                break

        if found is None:
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
