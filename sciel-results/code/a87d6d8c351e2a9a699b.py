import numpy as np
from itertools import combinations
import time

def construct():
    """
    Construct a 5-coloring of K_n with no monochromatic C4, maximizing n.
    Uses a deterministic local search / greedy construction approach.
    """
    n_max = 29
    n = 1
    # colors[i][j] for i < j, stored in a list or 2D array
    # We'll use a 2D array of size n_max x n_max, initialized to -1
    color = np.full((n_max, n_max), -1, dtype=np.int8)
    
    # Helper to check if adding edge (u, v) with color c creates a C4 in color c
    def is_valid_edge(u, v, c):
        # For C4: we need two vertices a, b such that both a and b are connected to u and v with color c
        # i.e., color[a][u] == c and color[a][v] == c, AND color[b][u] == c and color[b][v] == c
        # Wait, that's not right.
        # A C4 in color c means there exist distinct vertices u, v, a, b such that
        # (u,a), (a,v), (v,b), (b,u) are all color c.
        # When adding edge (u,v) with color c, we check if there exist a, b such that:
        # color[a][u] == c, color[a][v] == c is NOT the condition.
        # The condition is: there exist a, b distinct from u,v and from each other
        # such that color[a][u] == c, color[a][v] == c... no.
        
        # Let's re-derive.
        # We are adding edge (u, v) with color c.
        # A C4 containing (u, v) would look like: u - a - v - b - u
        # All edges must be color c.
        # So we need a, b such that:
        # color[a][u] == c
        # color[a][v] == c  <-- Wait, no. The edges are (u,a), (a,v), (v,b), (b,u).
        # So:
        # color[u][a] == c
        # color[a][v] == c
        # color[v][b] == c
        # color[b][u] == c
        
        # This means a is a common neighbor of u and v in color c? No.
        # a is connected to u with color c, and a is connected to v with color c.
        # So a is a common neighbor of u and v in color c.
        # b is connected to v with color c, and b is connected to u with color c.
        # So b is ALSO a common neighbor of u and v in color c.
        
        # So the condition for creating a C4 when adding (u,v) with color c is:
        # There exist at least TWO distinct vertices a, b (distinct from u, v)
        # such that color[a][u] == c AND color[a][v] == c.
        
        common_neighbours = []
        for x in range(n_max):
            if x == u or x == v:
                continue
            if color[x][u] == c and color[x][v] == c:
                # Only count if both edges are already set (not -1)
                if color[x][u] != -1 and color[x][v] != -1:
                    common_neighbours.append(x)
        
        return len(common_neighbours) < 2

    # Greedy construction: try to add vertices one by one
    for new_n in range(1, n_max + 1):
        found = False
        for attempt in range(100):  # Try a few different colorings for the new vertex
            # Try all 5! = 120 permutations of colors for the edges to new vertex
            # Or just try random colorings
            if new_n == 1:
                continue
            
            # Try to assign colors to edges (0, new_n-1), (1, new_n-1), ..., (new_n-2, new_n-1)
            # We can do this greedily or with backtracking
            # Since new_n is small initially, we can try all combinations or a simple greedy
            
            # Let's try a simple greedy assignment
            valid = True
            colors_for_new = []
            for i in range(new_n - 1):
                # Find a valid color for edge (i, new_n-1)
                valid_colors = []
                for c in range(5):
                    if is_valid_edge(i, new_n - 1, c):
                        valid_colors.append(c)
                
                if not valid_colors:
                    valid = False
                    break
                # Pick the first valid color (or random)
                # To add some diversity, we can shuffle
                np.random.seed(attempt + new_n)
                np.random.shuffle(valid_colors)
                chosen_c = valid_colors[0]
                colors_for_new.append(chosen_c)
            
            if valid:
                # Apply the colors
                for i, c in enumerate(colors_for_new):
                    color[i][new_n - 1] = c
                    color[new_n - 1][i] = c
                n = new_n
                found = True
                break
        
        if not found:
            break

    # If greedy failed early, try a more sophisticated local search
    # But for now, let's see how far we got.
    
    # Extract colors in lexicographic order
    colors_list = []
    for i in range(n):
        for j in range(i + 1, n):
            colors_list.append(int(color[i][j]))
    
    return {'n': n, 'colors': colors_list}
