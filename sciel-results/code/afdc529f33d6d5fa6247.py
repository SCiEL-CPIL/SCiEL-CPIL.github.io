import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    n = 29
    # Compute quadratic residues mod 29
    residues = set()
    for x in range(1, n):
        residues.add((x * x) % n)
    
    # Build color list
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in residues:
                colors.append(0)
            else:
                colors.append(1)
                
    # Validation: Check for K2,2,2 in either color
    # K2,2,2 has 12 edges. Its complement has 3 edges.
    # We check if any 6-subset induces a graph with 12 or 3 edges of color 0.
    
    # Precompute adjacency or just count edges on the fly?
    # With n=29, number of 6-subsets is ~475k.
    # Iterating over 6 vertices and checking 15 pairs is 475k * 15 = ~7M ops.
    # This is very fast in Python if optimized or even unoptimized.
    
    valid = True
    # To speed up, we can use a bitset or just simple loops.
    # Let's use a simple loop.
    
    # Map (i, j) to color index
    # index = i * (2*n - i - 1) // 2 + (j - i - 1) ? 
    # Easier: use a 2D array or set.
    # Since n is small, a 2D list is fine.
    adj = [[0]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[i][j] = c
            adj[j][i] = c
            idx += 1
            
    for combo in itertools.combinations(range(n), 6):
        edge_count_0 = 0
        for u, v in itertools.combinations(combo, 2):
            if adj[u][v] == 0:
                edge_count_0 += 1
        if edge_count_0 == 12 or edge_count_0 == 3:
            valid = False
            break
            
    if not valid:
        # Fallback to n=17 if 29 fails (should not happen)
        n = 17
        residues = set()
        for x in range(1, n):
            residues.add((x * x) % n)
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % n
                if diff in residues:
                    colors.append(0)
                else:
                    colors.append(1)
                    
    return {'n': n, 'colors': colors}
