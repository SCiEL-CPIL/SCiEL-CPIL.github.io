import random
import time

def construct():
    n = 15
    deadline = time.time() + 55
    
    # Base graph: K_{7,8}
    # Vertices 0..6 in A, 7..14 in B
    adj = [[0]*n for _ in range(n)]
    deg = [0]*n
    
    for i in range(7):
        for j in range(7, 15):
            adj[i][j] = 1
            adj[j][i] = 1
            deg[i] += 1
            deg[j] += 1
            
    # Add one edge in A: (0, 1)
    adj[0][1] = 1
    adj[1][0] = 1
    deg[0] += 1
    deg[1] += 1
    
    def is_valid(adj, deg):
        # Check if any simple path of length 3 has endpoints with equal degree
        # s - u - v - t
        # s != t
        # deg[s] != deg[t]
        
        # Precompute neighbors
        neighbors = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    neighbors[i].append(j)
                    
        for s in range(n):
            for t in range(s + 1, n):
                if deg[s] == deg[t]:
                    # Check if there is a path of length 3
                    found = False
                    for u in neighbors[s]:
                        if u == t: continue
                        for v in neighbors[u]:
                            if v == s or v == t: continue
                            if adj[v][t]:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        return False
        return True

    # Verify base
    if not is_valid(adj, deg):
        # Fallback to K_{7,8}
        adj[0][1] = 0
        adj[1][0] = 0
        deg[0] -= 1
        deg[1] -= 1
        
    best_adj = [row[:] for row in adj]
    best_deg = deg[:]
    best_edges = sum(sum(row) for row in adj) // 2
    
    # Local search to try to add more edges
    # Try adding any missing edge
    for restart in range(100):
        if time.time() > deadline:
            break
            
        improved = False
        # Try all missing edges
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
        random.shuffle(missing)
        
        for u, v in missing:
            # Try add
            adj[u][v] = 1
            adj[v][u] = 1
            deg[u] += 1
            deg[v] += 1
            
            if is_valid(adj, deg):
                improved = True
                best_adj = [row[:] for row in adj]
                best_deg = deg[:]
                best_edges += 1
            else:
                # Revert
                adj[u][v] = 0
                adj[v][u] = 0
                deg[u] -= 1
                deg[v] -= 1
                
            if time.time() > deadline:
                break

    # Convert to list of edges
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i][j]:
                result.append([i, j])
                
    return result
