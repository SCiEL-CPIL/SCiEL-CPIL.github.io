# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def canonical(e):
        a, b = e
        return (a, b) if a < b else (b, a)

    def hopcroft_karp(adj, n):
        """Maximum cardinality matching in a bipartite graph.
        adj: dict u -> set(v). Returns matchR: dict v -> u (for matched v)."""
        INF = float('inf')
        matchL = {}
        matchR = {}
        dist = {}

        def bfs():
            q = deque()
            for u in adj:
                if u not in matchL:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = INF
            found = False
            while q:
                u = q.popleft()
                for v in adj[u]:
                    mu = matchR.get(v)
                    if mu is None:
                        found = True
                    elif dist.get(mu, INF) == INF:
                        dist[mu] = dist[u] + 1
                        q.append(mu)
            return found

        def dfs(u):
            for v in adj[u]:
                mu = matchR.get(v)
                if mu is None or (dist.get(mu, INF) == dist[u] + 1 and dfs(mu)):
                    matchL[u] = v
                    matchR[v] = u
                    return True
            dist[u] = INF
            return False

        while bfs():
            for u in list(adj.keys()):
                if u not in matchL:
                    dfs(u)
        return matchR

    def build_2factor(adj):
        """Find a maximum 2-factor (max edge-disjoint cycles) in the graph
        using a maximum matching in the bipartite double cover.
        Returns list of cycles (each a list of distinct vertices, len>=3)."""
        bip_adj = {u: set() for u in adj}
        for u in adj:
            for v in adj[u]:
                bip_adj[u].add(v)
        matchR = hopcroft_karp(bip_adj, len(adj))

        factor = defaultdict(set)
        for v, u in matchR.items():
            factor[u].add(v)
            factor[v].add(u)

        cycles = []
        visited_v = set()
        for start in list(factor.keys()):
            if start in visited_v:
                continue
            if len(factor[start]) < 2:
                continue
            path = [start]
            prev = None
            cur = start
            while True:
                nxts = [x for x in factor[cur] if x != prev]
                if not nxts:
                    break
                nxt = nxts[0]
                if nxt == start:
                    if len(path) >= 3:
                        cycles.append(list(path))
                    break
                if nxt in path:
                    idx = path.index(nxt)
                    cyc = path[idx:]
                    if len(cyc) >= 3:
                        cycles.append(list(cyc))
                    break
                path.append(nxt)
                prev, cur = cur, nxt
            for x in path:
                visited_v.add(x)
        return cycles

    def heuristic_longest_cycle(adj, tries=150):
        """Randomized walk heuristic to find a long simple cycle."""
        if not adj:
            return None
        best = None
        # Use a deterministic seed based on adj to be reproducible within a run
        # but vary across different adj states
        seed_val = sum(hash(tuple(sorted(e))) for e in adj.keys()) & 0xffffffff
        rng = random.Random(seed_val)
        for _ in range(tries):
            cand = [u for u in adj if len(adj[u]) >= 2]
            if not cand:
                break
            start = rng.choice(cand)
            path = [start]
            used = {start}
            node = start
            while True:
                nbrs = [x for x in adj.get(node, ()) if x not in used]
                if not nbrs:
                    break
                # prefer closing to start if possible
                if start in nbrs and len(path) >= 2:
                    if best is None or len(path) > len(best):
                        best = list(path)
                    break
                # prefer neighbors with higher degree
                if len(nbrs) > 1:
                    nbrs.sort(key=lambda x: -len(adj.get(x, ())))
                    if rng.random() < 0.3:
                        node = rng.choice(nbrs)
                    else:
                        node = nbrs[0]
                else:
                    node = nbrs[0]
                path.append(node)
                used.add(node)
            if start in adj.get(node, ()) and len(path) >= 2:
                if best is None or len(path) > len(best):
                    best = list(path)
        if best and len(best) >= 3:
            return best
        return None

    def decompose(edges):
        if not edges:
            return []

        adj = defaultdict(set)
        remaining = set()
        for e in edges:
            c = canonical(e)
            remaining.add(c)
            adj[c[0]].add(c[1])
            adj[c[1]].add(c[0])

        pieces = []

        def remove_cycle(cyc):
            for i in range(len(cyc)):
                a, b = cyc[i], cyc[(i + 1) % len(cyc)]
                e = canonical((a, b))
                remaining.discard(e)
                adj[a].discard(b)
                adj[b].discard(a)
                if not adj[a]:
                    adj.pop(a, None)
                if not adj[b]:
                    adj.pop(b, None)

        def try_add_cycle(cyc):
            if not cyc or len(cyc) < 3:
                return False
            if len(set(cyc)) != len(cyc):
                return False
            for i in range(len(cyc)):
                a, b = cyc[i], cyc[(i + 1) % len(cyc)]
                if canonical((a, b)) not in remaining:
                    return False
            pieces.append(list(cyc))
            remove_cycle(cyc)
            return True

        # Phase 1: extract maximum 2-factor (edge-disjoint cycles).
        if time.time() - t0 < DEADLINE - 30:
            cycles = build_2factor(adj)
            for cyc in cycles:
                if time.time() - t0 > DEADLINE - 10:
                    break
                try_add_cycle(cyc)

        # Phase 2: aggressive greedy long-cycle packing on residual.
        while remaining:
            if time.time() - t0 > DEADLINE - 8:
                break
            cyc = heuristic_longest_cycle(adj, tries=120)
            if cyc:
                if try_add_cycle(cyc):
                    continue
            break

        # Phase 3: local improvement - merge remaining edges into triangles and small cycles.
        # Repeatedly scan for any cycle of length 3-6 in the residual.
        improved = True
        while improved and time.time() - t0 < DEADLINE - 4:
            improved = False
            if not remaining:
                break
            adj = defaultdict(set)
            for u, v in remaining:
                adj[u].add(v)
                adj[v].add(u)
            # Try to find any cycle of length 3 to 6
            found_cycle = None
            # Try triangles first (fast)
            for u in sorted(adj.keys()):
                if time.time() - t0 > DEADLINE - 4:
                    break
                nbrs_u = sorted(adj[u])
                for i in range(len(nbrs_u)):
                    for j in range(i + 1, len(nbrs_u)):
                        v, w = nbrs_u[i], nbrs_u[j]
                        if v in adj.get(w, set()):
                            # triangle u-v-w
                            if canonical((u, v)) in remaining and canonical((v, w)) in remaining and canonical((w, u)) in remaining:
                                found_cycle = [u, v, w]
                                break
                    if found_cycle:
                        break
                if found_cycle:
                    break
            if not found_cycle:
                # Try 4-cycles: u-v-w-x-u
                for u in sorted(adj.keys()):
                    if time.time() - t0 > DEADLINE - 4:
                        break
                    nbrs_u = sorted(adj[u])
                    for v in nbrs_u:
                        if len(adj.get(v, set())) < 2:
                            continue
                        for w in sorted(adj.get(v, set())):
                            if w == u:
                                continue
                            for x in sorted(adj.get(w, set())):
                                if x == u or x == v:
                                    continue
                                if u in adj.get(x, set()):
                                    # 4-cycle u-v-w-x
                                    if (canonical((u, v)) in remaining and canonical((v, w)) in remaining
                                            and canonical((w, x)) in remaining and canonical((x, u)) in remaining):
                                        found_cycle = [u, v, w, x]
                                        break
                            if found_cycle:
                                break
                        if found_cycle:
                            break
                    if found_cycle:
                        break
            if found_cycle:
                if try_add_cycle(found_cycle):
                    improved = True
                else:
                    break

        # Phase 4: any leftover edges as single edges.
        for e in list(remaining):
            pieces.append(list(e))
        remaining = set()

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        if time.time() - t0 > DEADLINE - 15:
            results.append([list(e) for e in edges])
        else:
            results.append(decompose(edges))

    return results
# EVOLVE-BLOCK-END
