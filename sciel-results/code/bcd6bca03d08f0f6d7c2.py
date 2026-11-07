"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_bicolored_forest(coloring, n, edges):
    """Check that for every pair of colors, the induced subgraph is a forest."""
    # Group vertices by color
    color_groups = defaultdict(list)
    for v in range(n):
        color_groups[coloring[v]].append(v)
    
    colors = list(color_groups.keys())
    
    # For each pair of colors, check if induced subgraph is a forest
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            c1, c2 = colors[i], colors[j]
            # Get vertices with these two colors
            vertices = set(color_groups[c1]) | set(color_groups[c2])
            
            # Build adjacency for this subgraph
            adj = defaultdict(list)
            for u, v in edges:
                if u in vertices and v in vertices:
                    adj[u].append(v)
                    adj[v].append(u)
            
            # Use union-find to detect cycles
            parent = {}
            rank = {}
            
            def find(x):
                if x not in parent:
                    parent[x] = x
                    rank[x] = 0
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False  # Cycle detected
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True
            
            has_cycle = False
            for u in vertices:
                for v in adj[u]:
                    if u < v:  # Avoid processing each edge twice
                        if not union(u, v):
                            has_cycle = True
                            break
                if has_cycle:
                    break
            
            if has_cycle:
                return False
    
    return True

def _check_adjacent_differs(coloring, n, edges):
    """Check that adjacent vertices have different colors."""
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    return True

def _is_valid(coloring, n, edges):
    """Check if a coloring is valid."""
    if not _check_adjacent_differs(coloring, n, edges):
        return False
    if not _check_bicolored_forest(coloring, n, edges):
        return False
    return True

def _count_distinct_colors(coloring):
    """Count the number of distinct colors used."""
    return len(set(coloring))

def construct():
    n = P['n']
    edges = P['edges']
    
    deadline = time.time() + 85  # Safety margin before 90s limit
    
    # Start with the trivial solution (all different colors)
    best_coloring = list(range(n))
    best_score = _count_distinct_colors(best_coloring)
    
    # Try to find a better solution with fewer colors
    # Strategy: try k colors for k = 4, 5, 6, ... and see if we can find a valid coloring
    
    for k in range(4, n):
        if time.time() > deadline:
            break
        
        # Try multiple random attempts for this k
        attempts = 0
        max_attempts = 1000
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Randomly assign colors from 0 to k-1
            coloring = [random.randint(0, k - 1) for _ in range(n)]
            
            # Check if valid
            if _is_valid(coloring, n, edges):
                score = _count_distinct_colors(coloring)
                if score < best_score:
                    best_coloring = coloring
                    best_score = score
                    if k == best_score:  # Can't do better than k colors with k colors
                        break
        
        # If we found a valid coloring with k colors, we can stop
        # (since we're trying in increasing order of k)
        if best_score <= k:
            break
    
    # Ensure we return Python ints in a list
    return [int(c) for c in best_coloring]
# EVOLVE-BLOCK-END
