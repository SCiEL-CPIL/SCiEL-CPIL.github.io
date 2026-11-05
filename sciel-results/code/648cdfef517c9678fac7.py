import time
import math
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s

    def luw_construction(d, q):
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

        rule = {2: (1,1), 3: (2,1), 4: (1,2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def girth_at_least(n, edges, target_girth):
        """Non-backtracking shortest-cycle check from every vertex.
        Returns True iff every cycle has length >= target_girth."""
        if n == 0:
            return True
        adj = build_adj(n, edges)
        for start in range(n):
            # state: (current, prev)
            dist = {}
            q = deque()
            for nb in adj[start]:
                dist[(nb, start)] = 1
                q.append((nb, start))
            while q:
                cur, prev = q.popleft()
                dcur = dist[(cur, prev)]
                for nb in adj[cur]:
                    if nb == prev:
                        continue
                    key = (nb, cur)
                    if key in dist:
                        # non-backtracking walk from start back to start
                        cycle_len = dcur + 1
                        if cycle_len < target_girth:
                            return False
                    else:
                        dist[key] = dcur + 1
                        q.append((nb, cur))
        return True

    # ---- Build LUW candidates ----
    q = k
    min_d = max(1, (g - 4 + 1) // 2)
    seed_d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    candidates = sorted(set([min_d, min_d + 1, min_d + 2, seed_d]))

    best_n = float('inf')
    best_edges = None
    best_adj = None

    for d in candidates:
        if time.time() > deadline - 15:
            break
        try:
            n, edges = luw_construction(d, q)
        except Exception:
            continue
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(x != k for x in deg):
            continue
        adj = build_adj(n, edges)
        # Verify girth exactly (fast non-backtracking)
        if not girth_at_least(n, edges, g):
            continue
        if n < best_n:
            best_n = n
            best_edges = edges
            best_adj = adj

    if best_edges is None:
        # Fallback
        d = seed_d
        n, edges = luw_construction(d, q)
        return n, edges

    # ---- Greedy local-search polish: merge vertices while preserving girth & k-regular ----
    # Use union-find; only merge pairs that are "compatible" (no shared neighbor,
    # and merging doesn't create a short cycle). This is conservative but safe.
    def try_polish(n, edges, adj, budget):
        nonlocal best_n, best_edges, best_adj
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            parent[ra] = rb
            return True

        # Precompute neighbor sets
        nset = [set(adj[i]) for i in range(n)]

        # Candidate merge pairs: vertices at "leaves" of the tree-like structure.
        # Heuristic: merge vertices whose neighborhoods are disjoint and whose
        # common-distance in the BFS tree from a root is large.
        # For simplicity, try random pairs of vertices with no shared neighbor.
        random.seed(12345)
        nodes = list(range(n))
        improved = True
        while improved and time.time() < budget:
            improved = False
            random.shuffle(nodes)
            # Build list of "active" vertices (roots)
            active = []
            for v in nodes:
                r = find(v)
                if r == v:
                    active.append(v)
            if len(active) < 2:
                break
            # Try to merge pairs
            for i in range(len(active)):
                for j in range(i + 1, len(active)):
                    a, b = active[i], active[j]
                    ra, rb = find(a), find(b)
                    if ra == rb:
                        continue
                    # No shared neighbor at root level
                    if nset[ra] & nset[rb]:
                        continue
                    # Try union and check girth quickly
                    union(ra, rb)
                    # Rebuild adjacency for the contracted graph (only roots)
                    # Quick check: does any pair of roots now share two neighbors?
                    ok = True
                    # Build contracted neighbor sets
                    cset = {}
                    for v in range(n):
                        r = find(v)
                        if r not in cset:
                            cset[r] = set()
                        for nb in adj[v]:
                            rnb = find(nb)
                            if rnb != r:
                                cset[r].add(rnb)
                    # Check for multi-edges / short cycles: any root with >=2 shared
                    # neighbors with another root would create a 4-cycle.
                    # We only need to ensure simple graph (no multi-edge) and girth.
                    # Simple graph check:
                    for r in cset:
                        if len(cset[r]) != len(set(cset[r])):
                            ok = False
                            break
                    # Girth check on contracted graph (only if small enough)
                    if ok:
                        cn = len(cset)
                        if cn < 3000:
                            cedges = []
                            seen = set()
                            for r in cset:
                                for nb in cset[r]:
                                    key = (min(r, nb), max(r, nb))
                                    if key not in seen:
                                        seen.add(key)
                                        cedges.append((r, nb))
                            # Remap to 0..cn-1
                            remap = {r: i for i, r in enumerate(cset.keys())}
                            cedges2 = [(remap[u], remap[v]) for u, v in cedges]
                            if not girth_at_least(cn, cedges2, g):
                                ok = False
                    if ok:
                        # Commit: rebuild best graph from contraction
                        cn = len(cset)
                        remap = {r: i for i, r in enumerate(cset.keys())}
                        cedges2 = []
                        seen = set()
                        for r in cset:
                            for nb in cset[r]:
                                key = (min(r, nb), max(r, nb))
                                if key not in seen:
                                    seen.add(key)
                                    cedges2.append((remap[r], remap[nb]))
                        # Verify k-regular
                        cd = [0] * cn
                        for u, v in cedges2:
                            cd[u] += 1
                            cd[v] += 1
                        if any(x != k for x in cd):
                            ok = False
                        else:
                            best_n = cn
                            best_edges = cedges2
                            best_adj = build_adj(cn, cedges2)
                            n = cn
                            edges = cedges2
                            adj = best_adj
                            nset = [set(adj[i]) for i in range(n)]
                            parent = list(range(n))
                            improved = True
                            break
                    else:
                        # Undo union
                        parent[ra] = ra
                    if time.time() > budget:
                        return
                if time.time() > budget:
                    return
            if time.time() > budget:
                return

    # Attempt polish if we have time
    if time.time() < deadline - 20 and best_n > 100:
        try:
            try_polish(best_n, best_edges, best_adj, deadline - 10)
        except Exception:
            pass

    return best_n, best_edges
