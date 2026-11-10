"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _is_3_colorable(adj, n):
    color = [-1] * n
    degrees = [bin(adj[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda v: -degrees[v])
    def dfs(idx):
        if idx == n:
            return True
        v = order[idx]
        used = 0
        for u in range(n):
            if color[u] >= 0 and (adj[v] >> u) & 1:
                used |= (1 << color[u])
        for c in range(3):
            if not (used >> c) & 1:
                color[v] = c
                if dfs(idx + 1):
                    return True
                color[v] = -1
        return False
    return dfs(0)

def construct():
    start_time = time.time()
    deadline = start_time + 145
    random.seed(42)
    n = 16
    adj = [0] * n
    
    def has_edge(i, j):
        return (adj[i] >> j) & 1
    def add_edge(i, j):
        adj[i] |= (1 << j)
        adj[j] |= (1 << i)
    def remove_edge(i, j):
        adj[i] &= ~(1 << j)
        adj[j] &= ~(1 << i)
    def edge_count():
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2
    def build_grotzsch():
        nonlocal adj
        adj = [0] * n
        for i in range(5):
            add_edge(i, (i+1) % 5)
        for i in range(5):
            for nb in [(i-1)%5, (i+1)%5]:
                add_edge(5+i, nb)
        for i in range(5):
            add_edge(10, 5+i)
    def greedy_densify():
        nonlocal adj
        progress = True
        while progress and time.time() < deadline:
            progress = False
            for i in range(n):
                for j in range(i+1, n):
                    if time.time() > deadline:
                        return
                    if has_edge(i, j) or (adj[i] & adj[j]):
                        continue
                    add_edge(i, j)
                    if not _is_3_colorable(adj, n):
                        progress = True
                    else:
                        remove_edge(i, j)
    def local_search():
        nonlocal adj
        for _ in range(50):
            if time.time() > deadline:
                return
            edges = [(i, j) for i in range(n) for j in range(i+1, n) if has_edge(i, j)]
            if not edges:
                return
            i, j = random.choice(edges)
            remove_edge(i, j)
            added = 0
            for a in range(n):
                for b in range(a+1, n):
                    if time.time() > deadline:
                        break
                    if has_edge(a, b) or (adj[a] & adj[b]):
                        continue
                    add_edge(a, b)
                    if not _is_3_colorable(adj, n):
                        added += 1
                    else:
                        remove_edge(a, b)
            if added < 1:
                add_edge(i, j)
    
    best_edges = None
    best_count = 0
    for restart in range(30):
        if time.time() > deadline:
            break
        build_grotzsch()
        greedy_densify()
        local_search()
        if edge_count() > best_count:
            best_count = edge_count()
            best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if has_edge(i, j)]
    
    if best_edges is None:
        build_grotzsch()
        best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if has_edge(i, j)]
    return best_edges
# EVOLVE-BLOCK-END

