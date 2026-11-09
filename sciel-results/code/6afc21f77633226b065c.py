import time
import random
from itertools import combinations

def _find_q3(adj, n=16):
    verts = list(range(n))
    for A in combinations(verts, 4):
        A_set = frozenset(A)
        cands = []
        for b in verts:
            if b in A_set:
                continue
            deg = sum(1 for a in A if adj[a][b])
            if deg == 3:
                cands.append(b)
        if len(cands) < 4:
            continue
        for B in combinations(cands, 4):
            B_set = frozenset(B)
            ok = True
            for b in B:
                if sum(1 for a in A if adj[a][b]) != 3:
                    ok = False
                    break
            if not ok:
                continue
            non_edges = []
            for a in A:
                missing = [b for b in B if not adj[a][b]]
                if len(missing) != 1:
                    ok = False
                    break
                non_edges.append(missing[0])
            if not ok:
                continue
            if set(non_edges) != B_set:
                continue
            return (A, B)
    return None

def _edge_list(adj, n=16):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
    return edges

def _count_edges(adj, n=16):
    return sum(sum(row) for row in adj) // 2

def construct():
    start_time = time.time()
    deadline = start_time + 140
    n = 16
    random.seed(42)
    
    best_edges = 0
    best_adj = None
    
    def _greedy_reduce(adj):
        for it in range(500):
            if time.time() > deadline:
                break
            q3 = _find_q3(adj, n)
            if q3 is None:
                break
            A, B = q3
            q3_edges = [(a, b) for a in A for b in B if adj[a][b]]
            if not q3_edges:
                break
            e = random.choice(q3_edges)
            adj[e[0]][e[1]] = 0
            adj[e[1]][e[0]] = 0
    
    def _try_add_edges(adj):
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        for i, j in all_pairs:
            if time.time() > deadline:
                break
            if adj[i][j]:
                continue
            adj[i][j] = 1
            adj[j][i] = 1
            if _find_q3(adj, n) is None:
                cnt = _count_edges(adj, n)
                nonlocal best_edges, best_adj
                if cnt > best_edges:
                    best_edges = cnt
                    best_adj = [row[:] for row in adj]
            else:
                adj[i][j] = 0
                adj[j][i] = 0
    
    # First run
    adj = [[1]*n for _ in range(n)]
    for i in range(n):
        adj[i][i] = 0
    _greedy_reduce(adj)
    cnt = _count_edges(adj, n)
    if cnt > best_edges:
        best_edges = cnt
        best_adj = [row[:] for row in adj]
    _try_add_edges(adj)
    
    # Multiple restarts
    while time.time() < deadline:
        random.seed()
        adj = [[1]*n for _ in range(n)]
        for i in range(n):
            adj[i][i] = 0
        _greedy_reduce(adj)
        cnt = _count_edges(adj, n)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = [row[:] for row in adj]
        _try_add_edges(adj)
    
    if best_adj is None:
        return [[a, b] for a in range(3) for b in range(3, 16)]
    
    return _edge_list(best_adj, n)
