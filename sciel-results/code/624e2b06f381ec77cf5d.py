import random
import numpy as np

def construct():
    n = 20
    colors = range(20)
    
    # edge_color[u][v] = color, or -1 if no edge
    edge_color = [[-1] * n for _ in range(n)]
    
    def is_valid_add(u, v, c):
        # Check all paths u-x-y-v of length 3
        for x in range(n):
            if x == u or x == v:
                continue
            if edge_color[u][x] == -1:
                continue
            c_ux = edge_color[u][x]
            
            for y in range(n):
                if y == u or y == v or y == x:
                    continue
                if edge_color[x][y] == -1:
                    continue
                c_xy = edge_color[x][y]
                
                if edge_color[y][v] == -1:
                    continue
                c_yv = edge_color[y][v]
                
                # Check if path colors are distinct
                if c_ux == c_xy or c_xy == c_yv or c_ux == c_yv:
                    return False # This path blocks any color
                
                # Check if new color c is distinct from path colors
                if c == c_ux or c == c_xy or c == c_yv:
                    return False
                    
        return True

    # To maximize edges, we try to add all edges.
    # Order of addition matters.
    # We can randomize the order of edges.
    
    edges_list = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Try multiple random orders to find a denser graph
    best_edges = []
    
    for _ in range(10): # 10 iterations
        random.shuffle(edges_list)
        
        current_edges = []
        edge_color = [[-1] * n for _ in range(n)]
        
        for u, v in edges_list:
            added = False
            # Try colors in random order or sorted?
            # Random might be better to avoid bias
            c_list = list(colors)
            random.shuffle(c_list)
            
            for c in c_list:
                if is_valid_add(u, v, c):
                    edge_color[u][v] = c
                    edge_color[v][u] = c
                    current_edges.append([u, v, c])
                    added = True
                    break
            
            # If not added, skip this edge
        
        if len(current_edges) > len(best_edges):
            best_edges = current_edges
            
    return best_edges
