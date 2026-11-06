# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
from random import Random

def construct():
    t0 = time.time()
    DEADLINE = 142.0  # safety margin before the 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def canonical(e):
        a, b = e
        return (a, b) if a < b else (b, a)

    def hopcroft_karp(left, right, adj):
        """Maximum bipartite matching. adj: dict left->list(right).
        Returns (matchL, matchR) where matchL[u]=v / matchR[v]=u (or -1)."""
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

    def extract_2factor_cycles(rem):
        """Maximum 2-matching -> edge-disjoint cycles (the 2-factor).
        Uses the bipartite double cover: matchL[u]=v (u->v) and matchR[u]=w (w->u).
        Trace the resulting directed 2-regular graph for cycles of length >= 3."""
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
        matchL, matchR = hopcroft_karp(left, right, bip)

        # directed successor: u -> matchL[u]
        visited = set()
        cycles = []
        for u in left:
            if u in visited or matchL[u] == -1:
                continue
            cyc = []
            cur = u
            while cur not in visited and matchL.get(cur, -1) != -1:
                visited.add(cur)
                cyc.append(cur)
                cur = matchL[cur]
            # close only if the walk returned to its start (a true directed cycle)
            if cur == u and len(cyc) >= 3:
                cycles.append(cyc)
        return cycles

    def cycle_edges(cyc):
        es = set()
        m = len(cyc)
        for i in range(m):
            es.add(canonical((cyc[i], cyc[(i + 1) % m])))
        return es

    def heuristic_longest_cycle(rem, adj, tries=140):
        """Degree-aware randomized DFS to find a long simple cycle."""
        if not adj:
            return None
        best = None
        rng = Random(hash(frozenset(rem)) & 0xffffffff)
        for _ in range(tries):
            cand = [u for u in adj if len(adj[u]) >= 2]
            if not cand:
                return best
            start = rng.choice(cand)
            path = [start]
            used = {start}
            node = start
            while True:
                nbrs = [x for x in adj.get(node, ()) if x not in used]
                if start in adj.get(node, ()) and len(path) >= 2:
                    # close the cycle
                    if best is None or len(path) > len(best):
                        best = list(path)
                    break
                if not nbrs:
                    break
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
        return best if (best and len(best) >= 3) else None

    def decompose(edges):
        if not edges:
            return []
        remaining = set(canonical(e) for e in edges)
        pieces = []

        # Phase 1: maximum 2-factor (globally optimal edge-disjoint cycles).
        if time.time() - t0 < DEADLINE - 20:
            for c in extract_2factor_cycles(remaining):
                ces = cycle_edges(c)
                if ces <= remaining and len(set(c)) == len(c):
                    pieces.append(list(c))
                    remaining -= ces

        # Phase 2: greedy long-cycle packing on the residual.
        while remaining:
            if time.time() - t0 > DEADLINE - 12:
                break
            adj = build_adj(remaining)
            cyc = heuristic_longest_cycle(remaining, adj, tries=100)
            if cyc and len(cyc) >= 3 and len(set(cyc)) == len(cyc):
                ces = cycle_edges(cyc)
                if ces <= remaining:
                    pieces.append(list(cyc))
                    remaining -= ces
                    continue
            break

        # Phase 3: local improvement -- fold any triangle into a 3-cycle piece.
        if time.time() - t0 < DEADLINE - 4:
            improved = True
            while improved and time.time() - t0 < DEADLINE - 4:
                improved = False
                adj = build_adj(remaining)
                if not adj:
                    break
                for u in sorted(adj.keys()):
                    if time.time() - t0 > DEADLINE - 4:
                        break
                    for v in sorted(adj[u]):
                        e1 = canonical((u, v))
                        if e1 not in remaining:
                            continue
                        for w in sorted(adj[v]):
                            if w == u or w == v:
                                continue
                            e2 = canonical((v, w))
                            e3 = canonical((u, w))
                            if e1 in remaining and e2 in remaining and e3 in remaining:
                                # fold triangle u-v-w into one cycle piece
                                pieces.append([u, v, w])
                                remaining.discard(e1)
                                remaining.discard(e2)
                                remaining.discard(e3)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break

        # Phase 4: any leftover edges become single-edge pieces.
        for e in remaining:
            pieces.append(list(e))
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
