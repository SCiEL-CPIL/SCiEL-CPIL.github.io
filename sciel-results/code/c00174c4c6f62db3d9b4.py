import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
N = 15

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin
    
    def check_validity(adj, deg):
        """Check if graph has no simple 3-edge path with equal-degree endpoints."""
        n = N
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    continue
                # For edge (u,v), find all s adjacent to u (s != v) and t adjacent to v (t != u)
                # such that s != t, s != v, t != u, and check deg(s) != deg(t)
                for s in range(n):
                    if s == u or s == v:
                        continue
                    if not adj[u][s]:
                        continue
                    for t in range(n):
                        if t == u or t == v or t == s:
                            continue
                        if not adj[v][t]:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True
    
    def count_edges(adj):
        count = 0
        for i in range(N):
            for j in range(i+1, N):
                if adj[i][j]:
                    count += 1
        return count
    
    def greedy_build(order):
        """Build graph greedily adding edges in given order if valid."""
        adj = [[False]*N for _ in range(N)]
        deg = [0]*N
        for u, v in order:
            # Check if adding edge (u,v) creates a violation
            adj[u][v] = adj[v][u] = True
            deg[u] += 1
            deg[v] += 1
            if not check_validity(adj, deg):
                adj[u][v] = adj[v][u] = False
                deg[u] -= 1
                deg[v] -= 1
        return adj, deg
    
    # Try multiple random orderings
    best_edges = 0
    best_adj = None
    
    all_possible_edges = [(i,j) for i in range(N) for j in range(i+1, N)]
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        random.shuffle(all_possible_edges)
        adj, deg = greedy_build(all_possible_edges)
        edges = count_edges(adj)
        if edges > best_edges:
            best_edges = edges
            best_adj = [row[:] for row in adj]
    
    # Local improvement: try to add edges
    if best_adj is not None:
        adj = [row[:] for row in best_adj]
        deg = [sum(adj[i][j] for j in range(N)) for i in range(N)]
        improved = True
        while improved:
            improved = False
            for u in range(N):
                for v in range(u+1, N):
                    if adj[u][v]:
                        continue
                    if time.time() > deadline:
                        break
                    # Try adding edge
                    adj[u][v] = adj[v][u] = True
                    deg[u] += 1
                    deg[v] += 1
                    if check_validity(adj, deg):
                        improved = True
                    else:
                        adj[u][v] = adj[v][u] = False
                        deg[u] -= 1
                        deg[v] -= 1
    
    # Build edge list
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if best_adj[i][j]:
                edges.append([i, j])
    
    return edges
# EVOLVE-BLOCK-END
