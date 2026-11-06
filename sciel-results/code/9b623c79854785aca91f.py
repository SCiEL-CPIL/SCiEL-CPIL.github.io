"""K3,3-free dense graph on 20 vertices with greedy + local search."""
import time
import itertools
import random

def construct():
    start_time = time.time()
    DEADLINE = 138.0
    n = 20
    
    all_triples = list(itertools.combinations(range(n), 3))
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    adj = [[False] * n for _ in range(n)]
    
    def add_edge(u, v):
        if u > v:
            u, v = v, u
        if not adj[u][v]:
            adj[u][v] = adj[v][u] = True
    
    def remove_edge(u, v):
        if u > v:
            u, v = v, u
        if adj[u][v]:
            adj[u][v] = adj[v][u] = False
    
    def check_add(u, v):
        if u == v or adj[u][v]:
            return False
        add_edge(u, v)
        ok = True
        for (a, b, c) in all_triples:
            cnt = 0
            for i in range(n):
                if i != a and i != b and i != c and adj[a][i] and adj[b][i] and adj[c][i]:
                    cnt += 1
                    if cnt >= 3:
                        ok = False
                        break
            if not ok:
                break
        remove_edge(u, v)
        return ok
    
    # Greedy construction
    for (u, v) in all_edges:
        if time.time() > DEADLINE:
            break
        if check_add(u, v):
            add_edge(u, v)
    
    best = sum(1 for i in range(n) for j in range(i + 1, n) if adj[i][j])
    best_list = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i][j]]
    
    # Local search with random restarts
    rng = random.Random(42)
    for restart in range(10):
        if time.time() > DEADLINE - 5:
            break
        for i in range(n):
            for j in range(n):
                adj[i][j] = False
        shuffled = all_edges[:]
        rng.shuffle(shuffled)
        for (u, v) in shuffled:
            if time.time() > DEADLINE - 5:
                break
            if check_add(u, v):
                add_edge(u, v)
        cnt = sum(1 for i in range(n) for j in range(i + 1, n) if adj[i][j])
        if cnt > best:
            best = cnt
            best_list = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i][j]]
    
    return best_list
