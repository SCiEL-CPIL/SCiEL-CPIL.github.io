import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140
    n = 20

    def can_add_edge(adj, u, v):
        if v in adj[u]:
            return True
        adj[u].add(v)
        adj[v].add(u)
        ok = True
        for x in range(n):
            if x == u or x == v:
                continue
            for y in range(x + 1, n):
                if y == u or y == v:
                    continue
                if len(adj[u] & adj[x] & adj[y]) >= 3:
                    ok = False
                    break
                if len(adj[v] & adj[x] & adj[y]) >= 3:
                    ok = False
                    break
            if not ok:
                break
        adj[u].discard(v)
        adj[v].discard(u)
        return ok

    def greedy_fill(adj):
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if v not in adj[u] and can_add_edge(adj, u, v):
                        adj[u].add(v)
                        adj[v].add(u)
                        improved = True
                    if time.time() > deadline - 5:
                        break
                if time.time() > deadline - 5:
                    break

    best_count = 0
    best_adj = [set() for _ in range(n)]

    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]

    for trial in range(200):
        if time.time() > deadline - 15:
            break
        adj = [set() for _ in range(n)]
        shuffled = all_edges[:]
        random.shuffle(shuffled)
        for u, v in shuffled:
            if can_add_edge(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
        greedy_fill(adj)
        count = sum(len(s) for s in adj) // 2
        if count > best_count:
            best_count = count
            best_adj = [set(s) for s in adj]

    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if v > u:
                edges.append([u, v])
    return edges
