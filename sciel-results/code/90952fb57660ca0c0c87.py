import time
import sys
from collections import deque

_START = time.time()
DEADLINE = _START + 145.0


def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _hopcroft_karp(adj, n_left, n_right):
    """
    Hopcroft-Karp maximum bipartite matching.
    adj: list of lists, adj[u] = list of v's in right set
    n_left, n_right: sizes of left and right sets
    Returns (matching_left, matching_right) where matching_left[u] = v or -1
    """
    INF = float('inf')
    pair_u = [-1] * n_left
    pair_v = [-1] * n_right
    dist = [0] * n_left

    def bfs():
        queue = deque()
        for u in range(n_left):
            if pair_u[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = INF
        found = False
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1:
                    found = True
                elif dist[pu] == INF:
                    dist[pu] = dist[u] + 1
                    queue.append(pu)
        return found

    def dfs(u):
        for v in adj[u]:
            pu = pair_v[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = INF
        return False

    matching = 0
    while bfs():
        for u in range(n_left):
            if pair_u[u] == -1:
                if dfs(u):
                    matching += 1
    return pair_u, pair_v


def _decompose(edges, n):
    """
    Decompose edge set into edge-disjoint cycles + leftover single edges.
    
    Strategy: Use maximum cycle packing via reduction to matching in a bipartite graph.
    For each edge (u,v), create two directed arcs (u->v) and (v->u).
    A cycle in the original graph corresponds to a cycle in the directed line graph.
    We can find edge-disjoint cycles by finding a maximum set of vertex-disjoint
    cycles in the directed line graph, which can be reduced to matching.
    
    However, a simpler and effective approach for small graphs:
    - Repeatedly find the longest cycle using a smart DFS
    - Use a global optimization pass at the end
    
    For these small graphs (n<=30, ~100 edges), we can use a more exhaustive approach:
    - Find all cycles up to a reasonable length
    - Use a greedy/set-cover style approach to maximize total cycle length
    """
    rem = {i: set() for i in range(n)}
    for u, v in edges:
        rem[u].add(v)
        rem[v].add(u)

    pieces = []

    def find_longest_cycle(start, max_depth=25):
        """Find the longest cycle starting from 'start' using DFS with pruning."""
        best = []
        
        # Use iterative DFS with explicit stack to avoid recursion issues
        # Stack entries: (node, parent, path_so_far)
        visited = set()
        visited.add(start)
        path = [start]
        
        # We'll do a DFS that tries to find cycles
        # To find the longest, we explore all possibilities but prune when we can't beat best
        
        def dfs(u, depth):
            nonlocal best
            if time.time() > DEADLINE:
                return
            if depth >= max_depth:
                return
            # If we can go back to start, we found a cycle
            if start in rem[u] and len(path) >= 3:
                if len(path) > len(best):
                    best = path[:]
                # Don't return, keep looking for longer cycles
            for v in sorted(rem[u], key=lambda x: -len(rem[x])):
                if v == path[-2] if len(path) >= 2 else False:
                    continue
                if v in visited:
                    continue
                if v == start:
                    # Found a cycle
                    if len(path) >= 3 and len(path) > len(best):
                        best = path[:]
                    continue
                visited.add(v)
                path.append(v)
                dfs(v, depth + 1)
                path.pop()
                visited.discard(v)
                if time.time() > DEADLINE:
                    return
        
        dfs(start, 0)
        return best if len(best) >= 3 else None

    def find_any_cycle():
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
                    if v in visited:
                        if v != parent.get(u, -1):
                            # Found a cycle
                            cyc = [u]
                            cur = u
                            while cur != v:
                                cur = parent[cur]
                                cyc.append(cur)
                            cyc.reverse()
                            return cyc
                    else:
                        parent[v] = u
                        visited.add(v)
                        queue.append(v)
        return None

    # Main loop: repeatedly find cycles and remove them
    # Use a multi-start strategy with preference for high-degree vertices
    iteration = 0
    while time.time() < DEADLINE - 5:
        iteration += 1
        degs = [(len(rem[i]), i) for i in range(n) if rem[i]]
        if not degs:
            break
        degs.sort(reverse=True)

        found = None
        best_len = 0

        # Try multiple starting points
        num_starts = min(6, len(degs))
        for idx in range(num_starts):
            if time.time() > DEADLINE - 5:
                break
            start = degs[idx][1]
            cyc = find_longest_cycle(start, max_depth=min(20, len(degs) + 2))
            if cyc is not None and len(cyc) > best_len:
                best_len = len(cyc)
                found = cyc
            if best_len >= 10:
                break

        if found is None:
            found = find_any_cycle()

        if found is None:
            break

        # Remove cycle edges
        for i in range(len(found)):
            a, b = found[i], found[(i + 1) % len(found)]
            rem[a].discard(b)
            rem[b].discard(a)
        pieces.append(found)

    # Leftover edges
    leftover = set()
    for u in range(n):
        for v in rem[u]:
            if u < v:
                leftover.add((u, v))
    for u, v in sorted(leftover):
        pieces.append([u, v])

    return pieces


def _decompose_optimal(edges, n):
    """
    More careful decomposition using a two-phase approach:
    Phase 1: Find a good set of cycles greedily
    Phase 2: Try to improve by swapping cycles
    
    For small graphs, we can afford a more exhaustive search.
    """
    rem = {i: set() for i in range(n)}
    for u, v in edges:
        rem[u].add(v)
        rem[v].add(u)

    pieces = []

    def find_cycle_from(start):
        """Find a long cycle starting from start using DFS."""
        best = []
        visited = set([start])
        path = [start]
        
        def dfs(u):
            nonlocal best
            if time.time() > DEADLINE:
                return
            if start in rem[u] and len(path) >= 3:
                if len(path) > len(best):
                    best = path[:]
            for v in sorted(rem[u], key=lambda x: -len(rem[x])):
                if v in visited:
                    continue
                if v == start:
                    if len(path) >= 3 and len(path) > len(best):
                        best = path[:]
                    continue
                visited.add(v)
                path.append(v)
                dfs(v)
                path.pop()
                visited.discard(v)
                if time.time() > DEADLINE:
                    return
        
        dfs(start)
        return best if len(best) >= 3 else None

    def find_any_cycle():
        for start in range(n):
            if not rem[start]:
                continue
            parent = {start: -1}
            visited = {start}
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in list(rem[u]):
                    if v in visited:
                        if v != parent.get(u, -1):
                            cyc = [u]
                            cur = u
                            while cur != v:
                                cur = parent[cur]
                                cyc.append(cur)
                            cyc.reverse()
                            return cyc
                    else:
                        parent[v] = u
                        visited.add(v)
                        queue.append(v)
        return None

    while time.time() < DEADLINE - 3:
        degs = [(len(rem[i]), i) for i in range(n) if rem[i]]
        if not degs:
            break
        degs.sort(reverse=True)

        found = None
        best_len = 0
        for idx in range(min(5, len(degs))):
            if time.time() > DEADLINE - 3:
                break
            start = degs[idx][1]
            cyc = find_cycle_from(start)
            if cyc is not None and len(cyc) > best_len:
                best_len = len(cyc)
                found = cyc
            if best_len >= 8:
                break

        if found is None:
            found = find_any_cycle()

        if found is None:
            break

        for i in range(len(found)):
            a, b = found[i], found[(i + 1) % len(found)]
            rem[a].discard(b)
            rem[b].discard(a)
        pieces.append(found)

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
        if time.time() > DEADLINE - 10:
            # Not enough time, use a quick fallback
            n, p, seed = spec
            edges = _instance(n, p, seed)
            # Just return all edges as single-edge pieces
            results.append([[u, v] for u, v in edges])
            continue
        n, p, seed = spec
        edges = _instance(n, p, seed)
        pieces = _decompose(edges, n)
        results.append(pieces)
    return results
