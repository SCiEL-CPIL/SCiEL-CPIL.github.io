import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = 110.0
    
    best_edges = []
    best_count = 0
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for seed in range(200):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        
        # adjacency[u] = set of (v, color)
        adj = [set() for _ in range(n)]
        edge_colors = {} # (u, v) -> c with u < v
        
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            if time.time() > deadline:
                break
            
            found_color = None
            for c in range(20):
                # Tentatively add
                adj[u].add((v, c))
                adj[v].add((u, c))
                edge_colors[(u, v)] = c
                
                valid = True
                # Check 4-cycles u-v-x-y-u
                # x in adj[v] (excluding u)
                # y in adj[u] (excluding v)
                # (x, y) in edge_colors
                
                for x, cx in list(adj[v]):
                    if x == u:
                        continue
                    for y, cy in list(adj[u]):
                        if y == v or y == x:
                            continue
                        # Check if (x, y) is an edge
                        key_xy = (min(x, y), max(x, y))
                        if key_xy in edge_colors:
                            cx_y = edge_colors[key_xy]
                            # Colors: c (u,v), cx (v,x), cx_y (x,y), cy (y,u)
                            colors = [c, cx, cx_y, cy]
                            if len(set(colors)) < 4:
                                valid = False
                                break
                    if not valid:
                        break
                
                if valid:
                    found_color = c
                    break
                else:
                    # Remove tentative
                    adj[u].remove((v, c))
                    adj[v].remove((u, c))
                    del edge_colors[(u, v)]
            
            if found_color is None:
                pass # Edge not added
        
        current_count = len(edge_colors)
        if current_count > best_count:
            best_count = current_count
            best_edges = [[u, v, c] for (u, v), c in edge_colors.items()]
            
    return best_edges
