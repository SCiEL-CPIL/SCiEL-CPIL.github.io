import random
import time

def construct():
    n = 20
    deadline = time.time() + 110.0  # Safety margin
    
    def is_valid_add(adj, u, v):
        # Check if adding edge (u, v) keeps the graph K2,3-free
        # 1. Check pair (u, v): common neighbors must be <= 2
        common_uv = 0
        for w in range(n):
            if adj[u][w] and adj[v][w]:
                common_uv += 1
                if common_uv > 2:
                    return False
        
        # 2. Check pairs (w, v) for w in N(u)
        for w in range(n):
            if w == v: continue
            if adj[u][w]:
                # u is a common neighbor of w and v (since we are adding u-v and w-u exists)
                # Count other common neighbors of w and v
                count = 0
                for x in range(n):
                    if x == u: continue
                    if adj[w][x] and adj[v][x]:
                        count += 1
                        if count > 1:
                            return False
        
        # 3. Check pairs (w, u) for w in N(v)
        for w in range(n):
            if w == u: continue
            if adj[v][w]:
                # v is a common neighbor of w and u
                count = 0
                for x in range(n):
                    if x == v: continue
                    if adj[w][x] and adj[u][x]:
                        count += 1
                        if count > 1:
                            return False
        
        return True

    def count_edges(adj):
        return sum(sum(row) for row in adj) // 2

    best_edges = 0
    best_adj = None

    # Run multiple random restarts
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        adj = [[False] * n for _ in range(n)]
        
        # Generate random permutation of edges
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline:
                break
            if is_valid_add(adj, u, v):
                adj[u][v] = True
                adj[v][u] = True
        
        curr_edges = count_edges(adj)
        if curr_edges > best_edges:
            best_edges = curr_edges
            best_adj = [row[:] for row in adj]

    # If no edges found (unlikely), return empty or seed
    if best_adj is None:
        return []

    # Convert adjacency matrix to list of edges
    result_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i][j]:
                result_edges.append([i, j])
                
    return result_edges
