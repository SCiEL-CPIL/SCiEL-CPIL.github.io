"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bipartite_matching(n, avail, deadline):
    """Find perfect matching via DFS augmenting paths."""
    match_r = [-1] * n

    def try_kun(u, seen):
        for v in avail[u]:
            if u == v:
                continue
            if not seen[v]:
                seen[v] = True
                if match_r[v] == -1 or try_kun(match_r[v], seen):
                    match_r[v] = u
                    return True
        return False

    for u in range(n):
        if time.time() > deadline:
            return None
        seen = [False] * n
        if not try_kun(u, seen):
            return None
    return match_r

def _find_2factor(n, adj, used_edges, deadline, rng):
    """Find a 2-factor as two edge-disjoint perfect matchings."""
    avail = [set() for _ in range(n)]
    for i in range(n):
        for j in adj[i]:
            if i != j and frozenset((i, j)) not in used_edges:
                avail[i].add(j)
        for j in avail[i]:
            if i not in avail[j]:
                avail[j].add(i)

    for i in range(n):
        if len(avail[i]) < 2:
            return None

    m1 = _bipartite_matching(n, avail, deadline)
    if m1 is None:
        return None

    used_m1 = set()
    for v in range(n):
        u = m1[v]
        used_m1.add(frozenset((u, v)))

    avail2 = [set() for _ in range(n)]
    for i in range(n):
        for j in avail[i]:
            if frozenset((i, j)) not in used_m1:
                avail2[i].add(j)
        for j in avail2[i]:
            if i not in avail2[j]:
                avail2[j].add(i)

    m2 = _bipartite_matching(n, avail2, deadline)
    if m2 is None:
        return None

    factor = [set() for _ in range(n)]
    for v in range(n):
        u1 = m1[v]
        u2 = m2[v]
        factor[v].add(u1)
        factor[v].add(u2)
    for v in range(n):
        for u in factor[v]:
            if v not in factor[u]:
                factor[u].add(v)

    for i in range(n):
        if len(factor[i]) != 2:
            return None
        for j in factor[i]:
            if i not in factor[j]:
                return None

    visited = set()
    cycles = []
    for start in range(n):
        if start in visited:
            continue
        cycle = [start]
        visited.add(start)
        cur = start
        prev = -1
        while True:
            nxt = -1
            for u in factor[cur]:
                if u != prev:
                    nxt = u
                    break
            if nxt == -1 or nxt == start:
                break
            cycle.append(nxt)
            visited.add(nxt)
            prev = cur
            cur = nxt
            if cur == start:
                break
        if len(cycle) >= 3:
            cycles.append(cycle)
    if len(visited) != n:
        return None
    return cycles

def _merge_cycles(cycles, adj, n, deadline, rng):
    """Merge cycles via 2-opt edge swaps."""
    if len(cycles) <= 1:
        return cycles
    edge_set = set()
    for cyc in cycles:
        k = len(cyc)
        for i in range(k):
            edge_set.add(frozenset((cyc[i], cyc[(i+1) % k])))
    for _ in range(1000):
        if time.time() > deadline or len(cycles) == 1:
            break
        i, j = rng.sample(range(len(cycles)), 2)
        c1, c2 = cycles[i], cycles[j]
        merged = False
        for a in c1:
            ia = c1.index(a)
            b = c1[(ia+1) % len(c1)]
            for c in c2:
                ic = c2.index(c)
                d = c2[(ic+1) % len(c2)]
                if c in adj[a] and d in adj[b]:
                    if frozenset((a,c)) not in edge_set and frozenset((b,d)) not in edge_set:
                        edge_set.discard(frozenset((a,b)))
                        edge_set.discard(frozenset((c,d)))
                        edge_set.add(frozenset((a,c)))
                        edge_set.add(frozenset((b,d)))
                        merged = True
                        break
                if d in adj[a] and c in adj[b]:
                    if frozenset((a,d)) not in edge_set and frozenset((b,c)) not in edge_set:
                        edge_set.discard(frozenset((a,b)))
                        edge_set.discard(frozenset((c,d)))
                        edge_set.add(frozenset((a,d)))
                        edge_set.add(frozenset((b,c)))
                        merged = True
                        break
            if merged:
                break
        if not merged:
            continue
        visited = set()
        new_cycles = []
        for start in range(n):
            if start in visited:
                continue
            cycle = [start]
            visited.add(start)
            cur = start
            prev = -1
            while True:
                nxt = -1
                for u in adj[cur]:
                    if frozenset((cur, u)) in edge_set and u != prev:
                        nxt = u
                        break
                if nxt == -1 or nxt == start:
                    break
                cycle.append(nxt)
                visited.add(nxt)
                prev = cur
                cur = nxt
                if cur == start:
                    break
            if len(cycle) >= 3:
                new_cycles.append(cycle)
        cycles = new_cycles
    return cycles

def construct():
    start_time = time.time()
    deadline = start_time + 80
    n = P['n']
    edges = P['edges']
    adj = _build_adj(edges, n)
    rng = random.Random(42)
    cycles_found = []
    used_edges_global = set()

    for _ in range(3):
        if time.time() > deadline:
            break
        found = False
        for _ in range(500):
            if time.time() > deadline:
                break
            result = _find_2factor(n, adj, used_edges_global, deadline, rng)
            if result is None:
                continue
            merged = _merge_cycles(result, adj, n, deadline, rng)
            if len(merged) == 1 and len(merged[0]) == n:
                cyc = merged[0]
                valid = len(set(cyc)) == n and n >= 3
                if valid:
                    for i in range(n):
                        a, b = cyc[i], cyc[(i+1) % n]
                        if frozenset((a, b)) in used_edges_global:
                            valid = False
                            break
                if valid:
                    for i in range(n):
                        a, b = cyc[i], cyc[(i+1) % n]
                        used_edges_global.add(frozenset((a, b)))
                    cycles_found.append(list(cyc))
                    found = True
                    break
        if not found:
            break

    if not cycles_found:
        edge_set = {frozenset(e) for e in edges}
        for e1 in edges:
            for e2 in edges:
                v1, v2 = e1
                v3 = e2[1]
                if v1 != v3 and v2 != v3 and (e2[0] == v1 or e2[0] == v2):
                    if frozenset((v2, v3)) in edge_set:
                        cycles_found.append([v1, v2, v3])
                        break
            if cycles_found:
                break

    return [[int(v) for v in c] for c in cycles_found]
# EVOLVE-BLOCK-END
