import random
import time

def construct():
    n = 20
    best_edges = []
    start_time = time.time()
    timeout = 115  # Safety margin
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for attempt in range(100):
        if time.time() - start_time > timeout:
            break
        
        random.shuffle(all_edges)
        adj = [[False]*n for _ in range(n)]
        color = [[-1]*n for _ in range(n)]
        edges = []
        
        for u, v in all_edges:
            added = False
            # Try to find a color for (u, v)
            # We can try random order of colors or all colors
            # Trying all 20 colors is cheap
            for c in range(20):
                # Check if adding (u, v) with color c is valid
                valid = True
                # Iterate over potential 4-cycles u-v-a-b-u
                # Edges (v, a), (a, b), (b, u) must exist
                # a != u, a != v, b != u, b != v
                # a != b
                for a in range(n):
                    if a == u or a == v: continue
                    if not adj[v][a]: continue
                    for b in range(n):
                        if b == u or b == v or b == a: continue
                        if not adj[a][b]: continue
                        if not adj[b][u]: continue
                        # Cycle u-v-a-b-u exists
                        # Colors: c (new), color[v][a], color[a][b], color[b][u]
                        c_va = color[v][a]
                        c_ab = color[a][b]
                        c_bu = color[b][u]
                        # Check distinctness
                        if len({c, c_va, c_ab, c_bu}) < 4:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    # Add edge
                    adj[u][v] = True
                    adj[v][u] = True
                    color[u][v] = c
                    color[v][u] = c
                    edges.append([u, v, c])
                    added = True
                    break
            
        if len(edges) > len(best_edges):
            best_edges = edges

    return best_edges
