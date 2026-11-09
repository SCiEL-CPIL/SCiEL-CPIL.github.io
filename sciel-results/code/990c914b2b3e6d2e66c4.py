"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(added_edges):
    n = P['n']
    base = set()
    for e in P['edges']:
        base.add((min(e[0], e[1]), max(e[0], e[1])))
    all_edges = set(base)
    for e in added_edges:
        u, v = int(e[0]), int(e[1])
        u, v = min(u, v), max(u, v)
        if (u, v) in all_edges:
            return None
        all_edges.add((u, v))
    adj = [[] for _ in range(n)]
    for u, v in all_edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check(adj):
    n = P['n']
    if adj is None:
        return False
    # Triangle-free: for each vertex, no two neighbors are adjacent
    for u in range(n):
        nbrs = adj[u]
        nbr_set = set(nbrs)
        for v in nbrs:
            for w in adj[v]:
                if w in nbr_set and w != u:
                    return False
    # Connected + diameter via BFS from each vertex
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        q = [src]
        qi = 0
        while qi < len(q):
            node = q[qi]
            qi += 1
            if dist[node] >= 4:
                continue
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    q.append(nb)
        if any(d == -1 for d in dist):
            return False
    return True

def _valid(added_edges):
    return _check(_build_adj(added_edges))

def _greedy_dominating_set():
    """
    Greedy set-cover approach: treat each potential hub edge as covering
    vertices within distance 2 in the augmented graph. We use a simpler
    greedy: start with full bipartite candidate set, remove edges one by one
    if the graph remains valid.
    """
    n = 24
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # All even-odd non-base edges
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u % 2) != (v % 2) and (u, v) not in base_edges:
                candidates.append((u, v))
    
    # Start with all candidates
    current = [list(c) for c in candidates]
    current_set = set(candidates)
    
    # Try to remove edges one by one (greedy set cover reduction)
    improved = True
    while improved:
        improved = False
        for i in range(len(current)):
            test = [e for j, e in enumerate(current) if j != i]
            if _valid(test):
                current = test
                improved = True
                break
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill
    
    n = 24
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # All valid candidate edges (even-odd, not in base)
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u % 2) != (v % 2) and (u, v) not in base_edges:
                candidates.append((u, v))
    
    # Phase 1: Greedy dominating set / set cover reduction
    best = _greedy_dominating_set()
    best_len = len(best)
    
    # Phase 2: Local search - try swapping one edge for another
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        for i in range(len(best)):
            if (time.time() - start_time) > deadline:
                break
            removed = best[i]
            remaining = [e for j, e in enumerate(best) if j != i]
            rem_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in remaining)
            for cand in candidates:
                if cand in rem_set:
                    continue
                if cand == (min(removed[0], removed[1]), max(removed[0], removed[1])):
                    continue
                test = remaining + [[int(cand[0]), int(cand[1])]]
                if _valid(test):
                    best = test
                    improved = True
                    break
            if improved:
                break
    
    # Phase 3: Try to remove more edges (if local search reduced size)
    if (time.time() - start_time) < deadline:
        improved = True
        while improved:
            improved = False
            for i in range(len(best)):
                if (time.time() - start_time) > deadline:
                    break
                test = [e for j, e in enumerate(best) if j != i]
                if _valid(test):
                    best = test
                    improved = True
                    break
    
    # Ensure proper types
    result = [[int(e[0]), int(e[1])] for e in best]
    return result

# EVOLVE-BLOCK-END
