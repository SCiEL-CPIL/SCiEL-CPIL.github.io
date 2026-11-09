import random
import time

def construct():
    n = 20
    deadline = time.time() + 110  # 110s safety margin
    
    best_edges = 0
    best_adj = [[0]*n for _ in range(n)]
    
    def is_valid_add(u, v, adj):
        # Check if adding edge (u,v) creates a K3,3
        # K3,3 parts: {u, a1, a2} and {v, b1, b2}
        # Edges needed:
        # (u,v) - being added
        # (u,b1), (u,b2) - must exist
        # (a1,v), (a2,v) - must exist
        # (a1,b1), (a1,b2), (a2,b1), (a2,b2) - must exist
        
        Nv = [x for x in range(n) if adj[v][x] and x != u]
        Nu = [x for x in range(n) if adj[u][x] and x != v]
        
        # We need to find a1, a2 in Nv and b1, b2 in Nu
        # such that a1, a2 are connected to b1, b2
        
        # Optimize: For each pair a1, a2 in Nv, check common neighbors in Nu
        for i in range(len(Nv)):
            a1 = Nv[i]
            for j in range(i+1, len(Nv)):
                a2 = Nv[j]
                # Common neighbors of a1 and a2 that are in Nu
                count = 0
                for b in Nu:
                    if adj[a1][b] and adj[a2][b]:
                        count += 1
                        if count >= 2:
                            return False
        return True

    def greedy_random():
        adj = [[0]*n for _ in range(n)]
        edges = []
        # Generate all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if is_valid_add(u, v, adj):
                adj[u][v] = 1
                adj[v][u] = 1
                edges.append((u, v))
        return adj, len(edges)

    # Run multiple restarts
    while time.time() < deadline:
        adj, edge_count = greedy_random()
        if edge_count > best_edges:
            best_edges = edge_count
            best_adj = [row[:] for row in adj]
            # If we reach a high number, we might stop early or continue
            # For n=20, max is likely < 100. If we hit 90, it's good.
    
    # Convert best_adj to list of edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                result.append([i, j])
    return result
