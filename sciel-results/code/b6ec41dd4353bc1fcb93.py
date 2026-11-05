import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 20
    colors = list(range(n))
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    best_edge_count = 0
    best_edge_color = {}
    
    random.seed(42)
    
    for trial in range(200):
        if time.time() > deadline - 20:
            break
        
        edge_color = {}
        adj = [set() for _ in range(n)]
        color_adj = [defaultdict(set) for _ in range(n)]
        
        edges_order = all_edges[:]
        random.shuffle(edges_order)
        
        for (u, v) in edges_order:
            if time.time() > deadline - 15:
                break
            
            colors_to_try = colors[:]
            random.shuffle(colors_to_try)
            
            for c in colors_to_try:
                # Check if color c is valid for edge (u,v)
                # Need: no path v-w-x-u in G_c
                valid = True
                for w in color_adj[c][v]:
                    if w == u:
                        continue
                    for x in color_adj[c][w]:
                        if x == u or x == v:
                            continue
                        if u in color_adj[c][x]:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    edge_color[(u, v)] = c
                    adj[u].add(v)
                    adj[v].add(u)
                    color_adj[c][u].add(v)
                    color_adj[c][v].add(u)
                    break
        
        count = len(edge_color)
        if count > best_edge_count:
            best_edge_count = count
            best_edge_color = dict(edge_color)
        
        if best_edge_count >= 180:
            break
    
    result = []
    for (u, v), c in sorted(best_edge_color.items()):
        result.append([u, v, c])
    
    return result
