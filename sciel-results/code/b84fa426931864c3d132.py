"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_bicolored_forest(coloring, n, edges):
    """Check if for every pair of colors, the induced subgraph is a forest."""
    # Group vertices by color
    color_to_verts = defaultdict(list)
    for v in range(n):
        color_to_verts[coloring[v]].append(v)
    
    # For each pair of colors, check if induced subgraph is a forest
    colors = list(color_to_verts.keys())
    for i in range(len(colors)):
        for j in range(i+1, len(colors)):
            c1, c2 = colors[i], colors[j]
            verts = set(color_to_verts[c1] + color_to_verts[c2])
            # Count edges within this induced subgraph
            edge_count = 0
            for u, v in edges:
                if u in verts and v in verts:
                    edge_count += 1
            # A forest on k vertices has at most k-1 edges
            if edge_count > len(verts) - 1:
                return False
    return True

def _is_valid_partial(coloring, n, edges):
    """Check partial coloring for immediate violations."""
    # Check adjacent vertices have different colors (where both are colored)
    for u, v in edges:
        if coloring[u] != -1 and coloring[v] != -1 and coloring[u] == coloring[v]:
            return False
    # Check bicolored forest constraint for fully determined pairs
    color_to_verts = defaultdict(list)
    for v in range(n):
        if coloring[v] != -1:
            color_to_verts[coloring[v]].append(v)
    
    colors = list(color_to_verts.keys())
    for i in range(len(colors)):
        for j in range(i+1, len(colors)):
            c1, c2 = colors[i], colors[j]
            verts = set(color_to_verts[c1] + color_to_verts[c2])
            edge_count = 0
            for u, v in edges:
                if u in verts and v in verts:
                    edge_count += 1
            if edge_count > len(verts) - 1:
                return False
    return True

def _solve_with_max_colors(max_colors, n, edges, deadline):
    """Backtracking solver with max_colors constraint."""
    coloring = [-1] * n
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Order vertices by degree descending for better pruning
    order = sorted(range(n), key=lambda v: -len(adj[v]))
    
    def backtrack(idx):
        if time.time() > deadline:
            return None
        if idx == len(order):
            return coloring[:]
        
        v = order[idx]
        # Try each color
        for c in range(max_colors):
            # Check if color c is valid for vertex v
            valid = True
            for nb in adj[v]:
                if coloring[nb] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            coloring[v] = c
            if _is_valid_partial(coloring, n, edges):
                result = backtrack(idx + 1)
                if result is not None:
                    return result
            coloring[v] = -1
        
        return None
    
    return backtrack(0)

def construct():
    n = P['n']
    edges = P['edges']
    start_time = time.time()
    deadline = start_time + 75  # 75s safety margin
    
    # Try increasing number of colors
    for max_colors in range(1, 50):
        result = _solve_with_max_colors(max_colors, n, edges, deadline)
        if result is not None:
            # Convert to Python ints
            return [int(c) for c in result]
    
    # Fallback: return trivial solution
    return [int(i) for i in range(n)]
# EVOLVE-BLOCK-END

