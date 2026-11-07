import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = 15
    adj = [[False]*n for _ in range(n)]
    deg = [0]*n
    
    def add_edge(u, v):
        adj[u][v] = adj[v][u] = True
        deg[u] += 1
        deg[v] += 1
    
    def remove_edge(u, v):
        adj[u][v] = adj[v][u] = False
        deg[u] -= 1
        deg[v] -= 1
    
    def is_valid():
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    continue
                for s in range(n):
                    if s == v or not adj[u][s]:
                        continue
                    for t in range(n):
                        if t == u or t == s or not adj[v][t]:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True
    
    random.seed(42)
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.6:
                add_edge(i, j)
    
    for _ in range(100):
        if time.time() > deadline:
            break
        if is_valid():
            break
        removed = False
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    remove_edge(u, v)
                    if is_valid():
                        removed = True
                        break
                    else:
                        add_edge(u, v)
            if removed:
                break
        if not removed:
            break
    
    for _ in range(300):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]:
                    add_edge(i, j)
                    if is_valid():
                        improved = True
                    else:
                        remove_edge(i, j)
        if not improved:
            break
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
    
    return edges
