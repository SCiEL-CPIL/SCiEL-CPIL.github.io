"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import deque


def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def canon(e):
        a, b = e
        return (a, b) if a < b else (b, a)

    def hopcroft_karp(left, right, adj):
        """Maximum bipartite matching. adj: dict left -> list of right.
        Returns (matchL, matchR)."""
        matchL = {u: -1 for u in left}
        matchR = {v: -1 for v in right}

        def bfs():
            dist = {}
            q = deque()
            for u in left:
                if matchL[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = -1
            found = False
            while q:
                u = q.popleft()
                for v in adj.get(u, ()):
                    mu = matchR[v]
                    if mu == -1:
                        found = True
                    elif dist.get(mu, -1) == -1:
                        dist[mu] = dist[u] + 1
                        q.append(mu)
            return dist, found

        def dfs(u):
            for v in adj.get(u, ()):
                mu = matchR[v]
                if mu == -1 or (dist.get(mu, -1) == dist.get(u, -1) + 1 and dfs(mu)):
                    matchL[u] = v
                    matchR[v] = u
                    return True
            dist[u] = -1
            return False

        while True:
            dist, found = bfs()
            if not found:
                break
            for u in left:
                if matchL[u] == -1:
                    dfs(u)
        return matchL, matchR

    def build_adj(rem):
        adj = {}
        for u, v in rem:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        return adj

    def cycles_from_match(matchL, left):
        """Follow the permutation u -> matchL[u] to extract simple cycles."""
        visited = set()
        cycles = []
        for u in left:
            if u in visited or matchL[u] == -1:
                continue
            cyc = []
            cur = u
            while cur not in visited and cur != -1:
                visited.add(cur)
                cyc.append(cur)
                cur = matchL[cur]
            if len(cyc) >= 3:
                cycles.append(cyc)
        return cycles

    def cycle_edges(cyc):
        m = len(cyc)
        es = set()
        for i in range(m):
            es.add(canon((cyc[i], cyc[(i + 1) % m])))
        return es

    def max_cycle_packing(rem):
        """Maximum-cardinality edge-disjoint cycle packing via bipartite matching."""
        adj = build_adj(rem)
        if not adj:
            return []
        vertices = sorted(adj.keys())
        left = vertices
        right = vertices
        bip = {u: [] for u in left}
        for u in left:
            for v in adj.get(u, ()):
                if v != u:
                    bip[u].append(v)
        matchL, _ = hopcroft_karp(left, right, bip)
        cycles = cycles_from_match(matchL, left)
        # keep only cycles whose edges are all present in rem
        good = []
        for c in cycles:
            if cycle_edges(c) <= rem:
                good.append(c)
        return good

    def heuristic_longest_cycle(rem, adj, rng, tries=80):
        """Randomized walk heuristic to find a long simple cycle."""
        best = None
        cand = [u for u in adj if len(adj[u]) >= 2]
        if not cand:
            return None
        for _ in range(tries):
            start = rng.choice(cand)
            path = [start]
            used = {start}
            node = start
            while True:
                nbrs = [x for x in adj.get(node, ()) if x not in used]
                if not nbrs:
                    break
                if start in nbrs and len(path) >= 2:
                    if best is None or len(path) > len(best):
                        best = list(path)
                    break
                node = rng.choice(nbrs)
                path.append(node)
                used.add(node)
            if start in adj.get(node, ()) and len(path) >= 2:
                if best is None or len(path) > len(best):
                    best = list(path)
        return best if (best and len(best) >= 3) else None

    def find_longest_cycle_greedy(remaining, adj, rng, tries=200):
        """Find a long simple cycle using randomized greedy walks with backtracking."""
        best = None
        cand = [u for u in adj if len(adj[u]) >= 2]
        if not cand:
            return None
        for _ in range(tries):
            start = rng.choice(cand)
            path = [start]
            used = {start}
            node = start
            while True:
                nbrs = [x for x in adj.get(node, ()) if x not in used]
                if start in nbrs and len(path) >= 2:
                    if best is None or len(path) > len(best):
                        best = list(path)
                    break
                if not nbrs:
                    break
                # Prefer neighbors with fewer options to avoid dead ends
                nbrs.sort(key=lambda x: len([y for y in adj.get(x, ()) if y not in used]))
                # Pick from top few with some randomness
                top_k = min(3, len(nbrs))
                node = rng.choice(nbrs[:top_k])
                path.append(node)
                used.add(node)
            if start in adj.get(node, ()) and len(path) >= 2:
                if best is None or len(path) > len(best):
                    best = list(path)
        return best if (best and len(best) >= 3) else None

    def decompose(edges):
        if not edges:
            return []
        remaining = set(canon(e) for e in edges)
        pieces = []

        # Phase 1: maximum-cardinality edge-disjoint cycle packing
        if time.time() - t0 < DEADLINE - 10:
            cyc = max_cycle_packing(remaining)
            for c in cyc:
                ces = cycle_edges(c)
                if ces <= remaining:
                    pieces.append(list(c))
                    remaining -= ces

        # Phase 2: iteratively find longest cycles in the remainder
        rng = Random(42)
        guard = 0
        while remaining and guard < 500:
            guard += 1
            if time.time() - t0 > DEADLINE - 5:
                break
            adj = build_adj(remaining)
            # Try to find a cycle; use more tries if graph is still large
            tries = 150 if len(remaining) > 20 else 80
            cyc = find_longest_cycle_greedy(remaining, adj, rng, tries=tries)
            if cyc and len(cyc) >= 3:
                ces = cycle_edges(cyc)
                if ces <= remaining:
                    pieces.append(list(cyc))
                    remaining -= ces
                    continue
            break

        for e in sorted(remaining):
            pieces.append(list(e))
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() - t0 > DEADLINE - 10:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            edges = instance(*spec)
            results.append(decompose(edges))

    return results
# EVOLVE-BLOCK-END
