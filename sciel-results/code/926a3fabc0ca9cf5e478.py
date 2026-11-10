"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def has_bicolored_cycle(coloring, edges):
    """Check if any pair of colors forms a cycle using union-find."""
    n = len(coloring)
    from collections import defaultdict
    color_pairs = defaultdict(list)
    for u, v in edges:
        cu, cv = coloring[u], coloring[v]
        if cu != cv:
            key = (min(cu, cv), max(cu, cv))
            color_pairs[key].append((u, v))
    
    for (c1, c2), edge_list in color_pairs.items():
        parent = {}
        def find(x):
            if x not in parent:
                parent[x] = x
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True
        for u, v in edge_list:
            if not union(u, v):
                return True
    return False

def count_colors(coloring):
    return len(set(coloring))

def is_valid(coloring, edges):
    # Check proper coloring
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    # Check bicolored forest
    if has_bicolored_cycle(coloring, edges):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Try deterministic patterns first
    # Pattern: i % k
    for k in range(2, 10):
        if time.time() > deadline:
            break
        coloring = [i % k for i in range(n)]
        if is_valid(coloring, edges):
            return coloring
            
    # Try pattern: (i // 2) % k + (i % 2) * k ? No, just try other mods
    # Try pattern: (i * 2) % k
    for k in range(2, 10):
        if time.time() > deadline:
            break
        coloring = [(i * 2) % k for i in range(n)]
        if is_valid(coloring, edges):
            return coloring
            
    # Fallback: Simple greedy with backtracking (limited)
    # Or just return a valid but high-color solution if time is low
    # A trivial valid solution is all distinct colors (50 colors)
    # But we want to minimize.
    
    # If we reach here, we try a quick local search
    # Start with random or mod 5
    best_coloring = [i % 5 for i in range(n)]
    if not is_valid(best_coloring, edges):
        # Try to fix
        # For simplicity and speed, if deterministic fails, return mod 10 or 50
        # Mod 50 is always valid (no edges between same color, bicolored subgraphs are matchings)
        best_coloring = list(range(n))
        
    return best_coloring

# EVOLVE-BLOCK-END
