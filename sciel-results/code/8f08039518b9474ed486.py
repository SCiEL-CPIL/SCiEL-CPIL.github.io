import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 15
    
    def count_edges(adj):
        return sum(sum(row) for row in adj) // 2
    
    def is_valid_fast(adj, degs):
        # Precompute for speed
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    continue
                # s neighbors of u (excluding v)
                for s in range(n):
                    if s == u or s == v or not adj[u][s]:
                        continue
                    ds = degs[s]
                    # t neighbors of v (excluding u, s)
                    for t in range(n):
                        if t == u or t == v or t == s or not adj[v][t]:
                            continue
                        if ds == degs[t]:
                            return False
        return True
    
    def full_check(adj):
        degs = [sum(adj[i]) for i in range(n)]
        return is_valid_fast(adj, degs)
    
    # Start with a good initial solution
    # Try: connect i-j if i+j >= 14
    adj = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if i + j >= 14:
                adj[i][j] = 1
                adj[j][i] = 1
    
    if full_check(adj):
        best_edges = count_edges(adj)
    else:
        best_edges = 0
        adj = [[0]*n for _ in range(n)]
    
    # Local search: try to add/remove/swap edges
    random.seed(123)
    
    for iteration in range(50000):
        if time.time() > deadline - 3:
            break
        
        # Try adding a random missing edge
        if random.random() < 0.7:
            while True:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u != v and not adj[u][v]:
                    break
            adj[u][v] = 1
            adj[v][u] = 1
            degs = [sum(adj[i]) for i in range(n)]
            if not is_valid_fast(adj, degs):
                adj[u][v] = 0
                adj[v][u] = 0
            else:
                ec = count_edges(adj)
                if ec > best_edges:
                    best_edges = ec
        else:
            # Try removing then re-adding (swap)
            while True:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u != v and adj[u][v]:
                    break
            adj[u][v] = 0
            adj[v][u] = 0
            degs = [sum(adj[i]) for i in range(n)]
            # Now try to add a different edge
            while True:
                a = random.randint(0, n-1)
                b = random.randint(0, n-1)
                if a != b and not adj[a][b]:
                    break
            adj[a][b] = 1
            adj[b][a] = 1
            degs2 = [sum(adj[i]) for i in range(n)]
            if not is_valid_fast(adj, degs2):
                # Revert
                adj[a][b] = 0
                adj[b][a] = 0
                adj[u][v] = 1
                adj[v][u] = 1
            else:
                ec = count_edges(adj)
                if ec > best_edges:
                    best_edges = ec
    
    # Build result
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
    
    return edges
