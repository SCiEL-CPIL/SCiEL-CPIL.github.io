import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 20
    best_edges = []
    best_count = 0
    
    # Pre-generate a list of all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_valid_addition(color_matrix, u, v, c):
        # Check if adding (u, v, c) creates a non-rainbow 4-cycle
        # A 4-cycle is u-w-x-v-u
        # Edges: (u,w), (w,x), (x,v) must exist.
        # Colours: c_uw, c_wx, c_xv, c must be distinct.
        
        for w in range(n):
            if w == u or w == v:
                continue
            c_uw = color_matrix[u][w]
            if c_uw == -1:
                continue
            
            for x in range(n):
                if x == u or x == v or x == w:
                    continue
                
                c_wx = color_matrix[w][x]
                if c_wx == -1:
                    continue
                
                c_xv = color_matrix[x][v]
                if c_xv == -1:
                    continue
                
                # Path u-w-x-v exists. Adding (v,u) completes cycle.
                # Check distinctness of {c, c_uw, c_wx, c_xv}
                if c == c_uw or c == c_wx or c == c_xv:
                    return False
                if c_uw == c_wx or c_uw == c_xv or c_wx == c_xv:
                    return False
                    
        return True

    def run_greedy(seed):
        random.seed(seed)
        color_matrix = [[-1 for _ in range(n)] for _ in range(n)]
        edges = []
        
        # Shuffle edges to randomize order
        edge_list = all_edges[:]
        random.shuffle(edge_list)
        
        for u, v in edge_list:
            # Try all colours in random order
            colours = list(range(n))
            random.shuffle(colours)
            
            for c in colours:
                if is_valid_addition(color_matrix, u, v, c):
                    color_matrix[u][v] = c
                    color_matrix[v][u] = c
                    edges.append([u, v, c])
                    break
            
        return edges

    seed_val = 42
    while time.time() < deadline:
        edges = run_greedy(seed_val)
        seed_val += 1
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges

    return best_edges
