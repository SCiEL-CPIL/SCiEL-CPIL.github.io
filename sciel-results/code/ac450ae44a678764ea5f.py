import random
import time

def construct():
    # Try to find a valid coloring for n=26, 27, 28, 29
    # Start with highest likely n
    for n in [26, 27, 28, 29]:
        # Try multiple random restarts
        for attempt in range(1000):
            if time.time() > 115:
                break
            colors = try_greedy(n)
            if colors is not None:
                return {'n': n, 'colors': colors}
    # Fallback to n=11 if all fail (should not happen)
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def try_greedy(n):
    # Adjacency lists for each color
    # neighbors[c][u] is a set of neighbors of u in color c
    neighbors = [ [set() for _ in range(n)] for _ in range(5) ]
    
    # List of all edges
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(edges)
    
    for u, v in edges:
        valid_color = -1
        # Try each color
        for c in range(5):
            # Check condition 1: |N_c(u) intersect N_c(v)| <= 1
            inter_uv = neighbors[c][u].intersection(neighbors[c][v])
            if len(inter_uv) > 1:
                continue
                
            # Check condition 2: For all w in N_c(v), N_c(u) intersect N_c(w) is empty
            # And for all w in N_c(u), N_c(v) intersect N_c(w) is empty
            
            # Optimization: 
            # If N_c(v) is empty, condition 2 is trivially true.
            # If N_c(u) is empty, condition 3 is trivially true.
            
            ok = True
            # Check w in N_c(v)
            for w in neighbors[c][v]:
                if not neighbors[c][u].isdisjoint(neighbors[c][w]):
                    ok = False
                    break
            if not ok:
                continue
            
            # Check w in N_c(u)
            for w in neighbors[c][u]:
                if not neighbors[c][v].isdisjoint(neighbors[c][w]):
                    ok = False
                    break
            if not ok:
                continue
                
            valid_color = c
            break
        
        if valid_color == -1:
            return None # Failed
            
        # Add edge (u, v) with color valid_color
        neighbors[valid_color][u].add(v)
        neighbors[valid_color][v].add(u)
        
    # Construct the output list
    # Lexicographic order: (0,1), (0,2)...(0,n-1), (1,2)...
    out_colors = []
    for i in range(n):
        for j in range(i+1, n):
            # Find the color of (i,j)
            # We can just check which neighbor set contains the other
            c_found = -1
            for c in range(5):
                if j in neighbors[c][i]:
                    c_found = c
                    break
            out_colors.append(c_found)
            
    return out_colors
