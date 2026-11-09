import random
import time
import sys

def construct():
    n = 15
    deadline = time.time() + 55  # 55 seconds limit
    
    def is_valid_incremental(adj, deg, u, v):
        # Check if adding edge (u, v) violates the constraint
        # 1. Check if (u, v) is already present
        if adj[u][v]:
            return False
        
        # Temporarily add edge
        adj[u][v] = 1
        adj[v][u] = 1
        deg[u] += 1
        deg[v] += 1
        
        # Check all pairs with equal degrees
        # Optimization: Only check pairs involving u or v, or pairs that were equal before?
        # Actually, degrees of u and v changed.
        # Pairs that might now be equal:
        # - (u, v) if deg[u] == deg[v]
        # - (u, w) where deg[w] == deg[u]
        # - (v, w) where deg[w] == deg[v]
        # - Pairs (w, z) where deg[w] == deg[z] and w != u, z != v.
        #   Did the path structure change for them?
        #   Adding edge (u, v) creates new paths of length 3.
        #   A new path s-a-b-t might exist now.
        #   If s and t had equal degrees BEFORE, and now there is a path between them, it's a violation.
        #   So we must check ALL equal-degree pairs for the existence of a path of length 3.
        
        # For n=15, checking all pairs is O(n^2 * n^2) = O(n^4) = 50k ops.
        # This is fast enough for many iterations.
        
        for s in range(n):
            for t in range(s + 1, n):
                if deg[s] == deg[t]:
                    if has_path3(adj, s, t):
                        # Violation
                        # Revert
                        adj[u][v] = 0
                        adj[v][u] = 0
                        deg[u] -= 1
                        deg[v] -= 1
                        return False
        
        # If no violation, keep the edge
        return True

    def has_path3(adj, s, t):
        # Check if there is a simple path of length 3 between s and t
        # s - u - v - t
        # u != t, v != s, u != v
        for u in range(n):
            if u == s or u == t:
                continue
            if adj[s][u]:
                for v in range(n):
                    if v == s or v == t or v == u:
                        continue
                    if adj[u][v] and adj[v][t]:
                        return True
        return False

    best_edges = 0
    best_adj = None
    best_deg = None
    
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        adj = [[0] * n for _ in range(n)]
        deg = [0] * n
        
        # Randomize edge order
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(edges)
        
        # Greedy addition
        for u, v in edges:
            if is_valid_incremental(adj, deg, u, v):
                # Edge added, is_valid_incremental updated adj and deg
                pass
            else:
                # Edge rejected, is_valid_incremental reverted adj and deg
                pass
        
        # Count edges
        edge_count = sum(sum(row) for row in adj) // 2
        if edge_count > best_edges:
            best_edges = edge_count
            best_adj = [row[:] for row in adj]
            best_deg = deg[:]
            
        # If we reached a high number, we can stop early or keep going
        if best_edges > 50:
             # Optional: break if we found a very good solution
             # But let's use the time to find even better
             pass

    # Convert best_adj to list of edges
    if best_adj is None:
        return []
        
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i][j]:
                result.append([i, j])
                
    return result
