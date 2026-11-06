import random
import time
import itertools

def construct():
    n = 20
    vertices = list(range(n))
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def can_add_edge(adj, u, v):
        # Check if adding (u,v) creates a K3,3
        # Potential K3,3: {u, a1, a2} x {v, b1, b2}
        # a1, a2 must be neighbors of v (excluding u)
        # b1, b2 must be neighbors of u (excluding v)
        # Edges (a1,b1), (a1,b2), (a2,b1), (a2,b2) must exist.
        
        N_v = [x for x in range(n) if adj[v][x] and x != u]
        N_u = [x for x in range(n) if adj[u][x] and x != v]
        
        if len(N_v) < 2 or len(N_u) < 2:
            return True
            
        for a1, a2 in itertools.combinations(N_v, 2):
            # Check if there are at least 2 common neighbors of a1 and a2 in N_u
            common_in_Nu = 0
            for b in N_u:
                if adj[a1][b] and adj[a2][b]:
                    common_in_Nu += 1
                    if common_in_Nu >= 2:
                        return False
        return True

    def greedy_addition(seed):
        random.seed(seed)
        adj = [[False] * n for _ in range(n)]
        for i in range(n):
            adj[i][i] = False
            
        edges_to_try = all_edges[:]
        random.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            if not adj[u][v]:
                if can_add_edge(adj, u, v):
                    adj[u][v] = True
                    adj[v][u] = True
                    
        return adj

    def local_search(adj):
        # Try to add any missing edge
        improved = True
        while improved:
            improved = False
            for u, v in all_edges:
                if not adj[u][v]:
                    if can_add_edge(adj, u, v):
                        adj[u][v] = True
                        adj[v][u] = True
                        improved = True
        return adj

    start_time = time.time()
    best_adj = None
    best_count = 0
    
    # Run multiple restarts
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() - start_time > 110:
            break
            
        seed = i * 12345 + 67890
        adj = greedy_addition(seed)
        adj = local_search(adj)
        
        count = sum(1 for i in range(n) for j in range(i+1, n) if adj[i][j])
        
        if count > best_count:
            best_count = count
            best_adj = [row[:] for row in adj]
            
    if best_adj is None:
        # Fallback to a simple construction if all failed (shouldn't happen)
        best_adj = [[False]*n for _ in range(n)]
        
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([i, j])
                
    return edges
