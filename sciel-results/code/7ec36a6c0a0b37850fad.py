"""Rewritten program: maximum-2-factor extraction (Hopcroft-Karp on line graph) + refinement."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct():
    t0 = time.time()
    DEADLINE = 148.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = set()
        for a, b in zip(order, order[1:]):
            edges.add((a, b) if a < b else (b, a))
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) not in edges and rng.random() < p:
                    edges.add((u, v))
        return sorted(edges)

    def decompose(edges):
        if not edges:
            return []
        rem = set()
        for u, v in edges:
            rem.add((u, v) if u < v else (v, u))

        def max_2factor(E):
            if not E:
                return []
            n = max(max(e) for e in E) + 1
            adj = [set() for _ in range(n)]
            for a, b in E:
                adj[a].add(b)
                adj[b].add(a)
            ladj = {}
            for a, b in E:
                for x in adj[a]:
                    if x != b:
                        ladj.setdefault((a, b), set()).add((x, b))
                for y in adj[b]:
                    if y != a:
                        ladj.setdefault((a, b), set()).add((a, y))
            keys = list(ladj.keys())
            idx = {e: i for i, e in enumerate(keys)}
            m = len(keys)
            L = [[idx[nb] for nb in s if nb in idx] for s in (ladj.get(e, set()) for e in keys)]

            pairU = [-1] * m
            pairV = [-1] * m
            dist = [0] * m
            while True:
                dq = deque()
                for u in range(m):
                    if pairU[u] == -1:
                        dist[u] = 0
                        dq.append(u)
                    else:
                        dist[u] = -1
                found_aug = False
                while dq:
                    u = dq.popleft()
                    for v in L[u]:
                        pu = pairV[v]
                        if pu == -1:
                            found_aug = True
                        elif dist[pu] == -1:
                            dist[pu] = dist[u] + 1
                            dq.append(pu)
                if not found_aug:
                    break
                def dfs(u):
                    for v in L[u]:
                        pu = pairV[v]
                        if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                            pairU[u] = v
                            pairV[v] = u
                            return True
                    dist[u] = -1
                    return False
                for u in range(m):
                    if pairU[u] == -1:
                        dfs(u)

            next_ptr = [-1] * m
            for u in range(m):
                if pairU[u] != -1:
                    v = pairU[u]
                    next_ptr[u] = v
                    next_ptr[v] = u
            used = [False] * m
            cycles = []
            for i in range(m):
                if not used[i] and pairU[i] != -1:
                    cycle_edges = []
                    cur = i
                    while not used[cur]:
                        used[cur] = True
                        cycle_edges.append(keys[cur])
                        cur = next_ptr[cur]
                    if len(cycle_edges) >= 3:
                        verts = [cycle_edges[0][0]]
                        for e in cycle_edges:
                            a, b = e
                            if a == verts[-1]:
                                verts.append(b)
                            else:
                                verts.append(a)
                        if verts[0] == verts[-1]:
                            verts.pop()
                        cycles.append(verts)
            return cycles

        pieces = []
        while rem:
            if time.time() - t0 > DEADLINE - 20:
                break
            cycles = max_2factor(rem)
            if not cycles:
                break
            for c in cycles:
                pieces.append(list(c))
                for i in range(len(c)):
                    a = c[i]; b = c[(i + 1) % len(c)]
                    rem.discard((a, b) if a < b else (b, a))
        for e in sorted(rem):
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
