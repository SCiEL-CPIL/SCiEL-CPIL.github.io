"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 105  # 15s safety margin before 120s limit
    
    n = 20
    best_edges = [[0, v, 0] for v in range(1, n)]  # Star as fallback (19 edges)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Random edge ordering
        all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_pairs)
        
        adj = [[-1]*n for _ in range(n)]
        edges = []
        
        for u, v in all_pairs:
            if time.time() > deadline:
                break
            for c in range(20):
                valid = True
                for x in range(n):
                    if x == u or x == v:
                        continue
                    if adj[v][x] == -1:
                        continue
                    for y in range(n):
                        if y == u or y == v or y == x:
                            continue
                        if adj[x][y] == -1 or adj[y][u] == -1:
                            continue
                        colors = [c, adj[v][x], adj[x][y], adj[y][u]]
                        if len(set(colors)) != 4:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    adj[u][v] = c
                    adj[v][u] = c
                    edges.append([u, v, c])
                    break
        
        if len(edges) > len(best_edges):
            best_edges = edges
    
    return best_edges
# EVOLVE-BLOCK-END
