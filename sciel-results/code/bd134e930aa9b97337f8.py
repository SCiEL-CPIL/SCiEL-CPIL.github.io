import time
import random

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 85
    
    def is_valid(adj, deg):
        for s in range(n):
            for t in range(s+1, n):
                if deg[s] != deg[t]:
                    continue
                for u in range(n):
                    if u == s or u == t:
                        continue
                    if not adj[s][u]:
                        continue
                    for v in range(n):
                        if v in (s, t, u):
                            continue
                        if adj[u][v] and adj[v][t]:
                            return False
        return True
    
    # Start with complete graph, remove edges to fix violations
    adj = [[0]*n for _ in range(n)]
    deg = [0]*n
    for i in range(n):
        for j in range(i+1, n):
            adj[i][j] = 1
            adj[j][i] = 1
            deg[i] += 1
            deg[j] += 1
    
    # Now iteratively remove edges that cause violations
    # Actually, let's do: try to add edges back after removing necessary ones
    
    # Better: start from empty, greedily add
    adj = [[0]*n for _ in range(n)]
    deg = [0]*n
    
    # Try all possible edge additions in a smart order
    # Sort potential edges by some criterion
    all_edges = [(i,j) for i in range(n) for j in range(i+1, n)]
    
    # Multiple restarts with different orderings
    best_edges = []
    best_count = 0
    
    for trial in range(30):
        if time.time() > deadline:
            break
        
        adj = [[0]*n for _ in range(n)]
        deg = [0]*n
        added = set()
        
        # Randomize order
        edges = all_edges[:]
        random.shuffle(edges)
        
        for i,j in edges:
            if time.time() > deadline:
                break
            adj[i][j] = 1
            adj[j][i] = 1
            deg[i] += 1
            deg[j] += 1
            if not is_valid(adj, deg):
                adj[i][j] = 0
                adj[j][i] = 0
                deg[i] -= 1
                deg[j] -= 1
        
        cnt = sum(sum(row) for row in adj) // 2
        if cnt > best_count:
            best_count = cnt
            best_adj = [row[:] for row in adj]
    
    # Local search: try to improve
    if time.time() < deadline - 5:
        adj = [row[:] for row in best_adj]
        deg = [sum(row) for row in adj]
        
        for iteration in range(5000):
            if time.time() > deadline - 5:
                break
            
            # Try removing a random edge and re-adding others
            # Or: try flipping a random edge
            i, j = random.randint(0, n-1), random.randint(0, n-1)
            if i == j:
                continue
            i, j = min(i,j), max(i,j)
            
            if adj[i][j]:
                # Try removing
                adj[i][j] = 0
                adj[j][i] = 0
                deg[i] -= 1
                deg[j] -= 1
                # Now try to add other edges
                improved = False
                for a in range(n):
                    if time.time() > deadline - 5:
                        break
                    for b in range(a+1, n):
                        if a == i or a == j or b == i or b == j:
                            continue
                        if not adj[a][b]:
                            adj[a][b] = 1
                            adj[b][a] = 1
                            deg[a] += 1
                            deg[b] += 1
                            if is_valid(adj, deg):
                                improved = True
                            else:
                                adj[a][b] = 0
                                adj[b][a] = 0
                                deg[a] -= 1
                                deg[b] -= 1
                if not improved:
                    # Revert the removal
                    adj[i][j] = 1
                    adj[j][i] = 1
                    deg[i] += 1
                    deg[j] += 1
            else:
                # Try adding
                adj[i][j] = 1
                adj[j][i] = 1
                deg[i] += 1
                deg[j] += 1
                if not is_valid(adj, deg):
                    adj[i][j] = 0
                    adj[j][i] = 0
                    deg[i] -= 1
                    deg[j] -= 1
    
    # Build result
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                result.append([int(i), int(j)])
    
    return result
