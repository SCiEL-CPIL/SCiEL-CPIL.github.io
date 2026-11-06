import time
import random

def construct():
    deadline = time.time() + 145
    best_n = 0
    best_edges = []

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def is_c5_free(n, edges):
        if n > 35:
            return False
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            if u == v:
                return False
            adj[u][v] = True
            adj[v][u] = True
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u][v]:
                    continue
                found = False
                neighbors_u = [x for x in range(n) if adj[u][x] and x != v]
                neighbors_v = [y for y in range(n) if adj[v][y] and y != u]
                for a in neighbors_u:
                    if found:
                        break
                    for c in neighbors_v:
                        if found:
                            break
                        if a == c:
                            continue
                        for b in range(n):
                            if b in (u, v, a, c):
                                continue
                            if adj[a][b] and adj[b][c]:
                                found = True
                                break
                if found:
                    return False
        return True

    def exact_alpha(n, edges):
        # Exact independence number via branch and bound
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        best = [0]
        
        def branch(candidates, current_size):
            if not candidates:
                if current_size > best[0]:
                    best[0] = current_size
                return
            if current_size + len(candidates) <= best[0]:
                return
            # Pick a vertex with max degree in candidates
            v = max(candidates, key=lambda x: len(adj[x] & set(candidates)))
            # Branch: include v
            new_cands = [x for x in candidates if x != v and x not in adj[v]]
            branch(new_cands, current_size + 1)
            # Branch: exclude v
            branch([x for x in candidates if x != v], current_size)
        
        branch(list(range(n)), 0)
        return best[0]

    def validate(n, edges):
        if len(edges) != len(set((min(u, v), max(u, v)) for u, v in edges)):
            return False
        for u, v in edges:
            if u == v or u < 0 or u >= n or v < 0 or v >= n:
                return False
        if not is_c5_free(n, edges):
            return False
        if exact_alpha(n, edges) >= 9:
            return False
        return True

    # Candidate n=32: 8 disjoint K4
    edges_32 = []
    for block in range(8):
        base = block * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges_32.append([base + i, base + j])
    if validate(32, edges_32):
        best_n = 32
        best_edges = edges_32

    # Candidate n=33: 8 K4s + vertex 32 connected to all of K4#0
    edges_33 = list(edges_32)
    for i in range(4):
        edges_33.append([32, i])
    if validate(33, edges_33):
        best_n = 33
        best_edges = edges_33

    # Candidate n=34: 8 K4s + vertices 32,33
    # 32 connected to all of K4#0, 33 connected to all of K4#1
    # Check: independent set containing 32 and 33: can pick 1 from each of K4#2..#7 = 6, total 8. 
    # Containing only 32: 1 from K4#1..#7 = 7 + 32 = 8. Good.
    # C5-free: cycles through 32 need path len 3 in K4#0, impossible. Similarly 33.
    # Cycle through both 32 and 33: 32-a-b-c-33, a in K4#0, c in K4#1. Need path a-b-c len 2, 
    # but a in K4#0, c in K4#1, no edges between K4#0 and K4#1, so b can't connect both. C5-free.
    edges_34 = list(edges_32)
    for i in range(4):
        edges_34.append([32, i])
    for i in range(4):
        edges_34.append([33, 4 + i])
    if validate(34, edges_34):
        best_n = 34
        best_edges = edges_34

    # Candidate n=35: 8 K4s + vertices 32,33,34
    # 32->K4#0, 33->K4#1, 34->K4#2
    # Independent set with all three: 1 from K4#3..#7 = 5 + 3 = 8. Good.
    # C5-free by same argument.
    edges_35 = list(edges_32)
    for i in range(4):
        edges_35.append([32, i])
    for i in range(4):
        edges_35.append([33, 4 + i])
    for i in range(4):
        edges_35.append([34, 8 + i])
    if validate(35, edges_35):
        best_n = 35
        best_edges = edges_35

    if best_n == 0:
        best_n = 32
        best_edges = edges_32

    return {'n': best_n, 'edges': best_edges}
