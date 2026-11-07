"""Solve the few-color bicolored-forest coloring on the pinned 4-regular graph.
Strategy:
  1. The graph is a Cayley graph on Z_50 (edges i <-> i+1, i+7, i+43 mod 50).
     For a linear coloring c(v) = (a*v) mod k, each edge step d becomes a
     bicolored subgraph on Z_k with step (a*d) mod k, which is a union of
     gcd(k, a*d) cycles of length k/gcd.  That subgraph is a forest iff
     gcd(k, a*d) == 1 (a single k-cycle => no cycle).  Hence k works iff
     gcd(k, a) == gcd(k, 7) == 1 and k >= 3.  The smallest such k is 6.
  2. Enumerate (k, a) in increasing k, verify each candidate with the exact
     union-find bicolored-cycle check, and return the first valid one.
  3. As a refinement / safety net, run a fast cached merge-based reduction
     on the best coloring found (it can only keep or improve validity).
"""
import time
from math import gcd
from collections import defaultdict, Counter

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _build_adj():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _fast_check(coloring):
    """Fast exact validity check (adjacency + bicolored forest) using cached
    per-color-pair edge lists.  Returns (is_valid, pair_edges)."""
    n = len(coloring)
    # Adjacency
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False, None
    # Group edges by color pair
    pair_edges = defaultdict(list)
    for u, v in P['edges']:
        cu, cv = coloring[u], coloring[v]
        key = (cu, cv) if cu < cv else (cv, cu)
        pair_edges[key].append((u, v))
    for (c1, c2), edges in pair_edges.items():
        parent = list(range(n))
        rank = [0] * n
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for u, v in edges:
            ru, rv = find(u), find(v)
            if ru == rv:
                return False, None
            if rank[ru] < rank[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            if rank[ru] == rank[rv]:
                rank[ru] += 1
    return True, pair_edges


def _try_reduce(coloring, adj, deadline):
    """Cached incremental merge-based reduction.  Returns best coloring."""
    n = len(coloring)
    valid, pair_edges = _fast_check(coloring)
    if not valid:
        return coloring

    # adjacency set for the quick "adjacent" pre-filter
    adj_set = [set(a) for a in adj]

    def rebuild(coloring):
        for u, v in P['edges']:
            if coloring[u] == coloring[v]:
                return False, None
        pe = defaultdict(list)
        for u, v in P['edges']:
            cu, cv = coloring[u], coloring[v]
            key = (cu, cv) if cu < cv else (cv, cu)
            pe[key].append((u, v))
        for (c1, c2), edges in pe.items():
            parent = list(range(n))
            rank = [0] * n
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            for u, v in edges:
                ru, rv = find(u), find(v)
                if ru == rv:
                    return False, None
                if rank[ru] < rank[rv]:
                    ru, rv = rv, ru
                parent[rv] = ru
                if rank[ru] == rank[rv]:
                    rank[ru] += 1
        return True, pe

    changed = True
    while changed and time.time() < deadline - 3:
        changed = False
        distinct = sorted(set(coloring))
        if len(distinct) <= 2:
            break
        counts = Counter(coloring)
        ordered = sorted(distinct, key=lambda c: (-counts[c], c))
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                ca, cb = ordered[i], ordered[j]
                # Quick filter: no edge directly between ca and cb
                if (ca, cb) not in pair_edges and (cb, ca) not in pair_edges:
                    new_coloring = [ca if c == cb else c for c in coloring]
                    ok, pe = rebuild(new_coloring)
                    if ok:
                        coloring = new_coloring
                        pair_edges = pe
                        changed = True
                        break
            if changed:
                break
    return coloring


def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    n = P['n']
    adj = _build_adj()
    
    # Try k=2, 3, 4, 5, 6 in order with linear coloring c(v) = (a*v) mod k
    for k in [2, 3, 4, 5, 6, 7, 8]:
        if time.time() > deadline - 5:
            break
        for a in range(1, k):
            if gcd(k, a) != 1:
                continue
            if gcd(k, 7) != 1:
                continue
            coloring = [(a * v) % k for v in range(n)]
            valid, _ = _fast_check(coloring)
            if valid:
                # Try to reduce further
                reduced = _try_reduce(coloring, adj, deadline)
                return [int(c) for c in reduced]
    
    # Fallback: use a structured coloring and local search
    # Try several initial colorings
    initial_colorings = [
        [v % 5 for v in range(n)],
        [(v // 10) * 2 + (v % 2) for v in range(n)],
        [(v % 4) + 4 * ((v // 10) % 2) for v in range(n)],
    ]
    
    best_result = None
    best_count = n + 1
    
    for init in initial_colorings:
        if time.time() > deadline - 5:
            break
        # Fix adjacency violations
        coloring = [int(c) for c in init]
        for _ in range(100000):
            violations = [(u, v) for u, v in P['edges'] if coloring[u] == coloring[v]]
            if not violations:
                break
            u, v = violations[0]
            used = set(coloring[nb] for nb in adj[u])
            avail = [c for c in range(10) if c not in used]
            if avail:
                coloring[u] = avail[0]
            else:
                used_v = set(coloring[nb] for nb in adj[v])
                avail_v = [c for c in range(10) if c not in used_v]
                if avail_v:
                    coloring[v] = avail_v[0]
        
        valid, _ = _fast_check(coloring)
        if not valid:
            continue
        
        cnt = len(set(coloring))
        if cnt < best_count:
            best_count = cnt
            best_result = [int(c) for c in coloring]
        
        # Try local search to reduce colors
        if time.time() > deadline - 10:
            break
        improved = _try_reduce(coloring, adj, deadline - 5)
        if improved is not None:
            cnt2 = len(set(improved))
            if cnt2 < best_count:
                best_count = cnt2
                best_result = improved
    
    if best_result is not None:
        return best_result
    
    # Guaranteed valid fallback: all distinct colors
    return list(range(n))


if __name__ == "__main__":
    result = construct()
    print("colors used:", len(set(result)))
    print("valid:", _fast_check(result)[0])
