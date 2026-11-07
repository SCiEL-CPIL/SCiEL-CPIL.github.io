"""Solve: minimum colors for degree-4 graph with bicolored-forest constraint."""
import time
from collections import defaultdict
import random

# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency():
    adj = defaultdict(list)
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_bicolored_forest(coloring, adj, c1, c2):
    """Check if edges between color c1 and c2 form a forest."""
    n = P['n']
    uf = list(range(n))
    rank = [0] * n
    
    def find(x):
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x
    
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        uf[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True
    
    for u in range(n):
        if coloring[u] == c1:
            for v in adj[u]:
                if coloring[v] == c2:
                    if not union(u, v):
                        return False
    return True

def _is_valid(coloring, adj):
    """Check all constraints."""
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False
    colors_used = set(coloring)
    for c1 in colors_used:
        for c2 in colors_used:
            if c1 >= c2:
                continue
            if not _check_bicolored_forest(coloring, adj, c1, c2):
                return False
    return True

def _count_colors(coloring):
    return len(set(coloring))

def _try_merge(coloring, adj, c1, c2):
    """Try to merge color class c2 into c1. Returns new coloring or None."""
    n = P['n']
    new_coloring = coloring[:]
    
    # Check adjacency: no vertex with color c1 should be adjacent to a vertex with color c2
    for u in range(n):
        if new_coloring[u] == c1:
            for v in adj[u]:
                if new_coloring[v] == c2:
                    return None  # Can't merge, adjacent vertices would have same color
    
    # Recolor all c2 to c1
    for u in range(n):
        if new_coloring[u] == c2:
            new_coloring[u] = c1
    
    # Check all bicolored forest constraints
    if not _is_valid(new_coloring, adj):
        return None
    
    return new_coloring

def _local_search(coloring, adj, deadline):
    """Try to reduce number of colors by merging classes."""
    n = P['n']
    current = coloring[:]
    best = current[:]
    best_count = _count_colors(best)
    
    while time.time() < deadline:
        colors_used = sorted(set(current))
        merged = False
        
        for i in range(len(colors_used)):
            for j in range(i + 1, len(colors_used)):
                c1, c2 = colors_used[i], colors_used[j]
                new_c = _try_merge(current, adj, c1, c2)
                if new_c is not None:
                    count = _count_colors(new_c)
                    if count < best_count:
                        best = new_c[:]
                        best_count = count
                    current = new_c
                    merged = True
                    break
            if merged:
                break
        
        if not merged:
            break
    
    return best, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline
    
    adj = _build_adjacency()
    n = P['n']
    
    # Start with a valid coloring: each vertex gets its own color
    # Then try to merge colors
    coloring = list(range(n))
    
    # First, try a greedy k-coloring for various k
    # Try to find a valid coloring with few colors
    
    best_coloring = list(range(n))
    best_count = n
    
    # Try greedy coloring with increasing k
    for k in range(2, n + 1):
        if time.time() > deadline - 10:
            break
        
        # Greedy coloring
        c = [-1] * n
        for v in range(n):
            neighbor_colors = set()
            for u in adj[v]:
                if c[u] != -1:
                    neighbor_colors.add(c[u])
            for color in range(k):
                if color not in neighbor_colors:
                    c[v] = color
                    break
            if c[v] == -1:
                break
        
        if c[-1] == -1:
            continue
        
        # Check validity
        if _is_valid(c, adj):
            count = _count_colors(c)
            if count < best_count:
                best_count = count
                best_coloring = c[:]
                break
        
        # Try local search to improve
        if time.time() < deadline - 5:
            improved, improved_count = _local_search(c, adj, deadline - 5)
            if improved_count < best_count:
                best_count = improved_count
                best_coloring = improved[:]
    
    # Ensure types are correct
    return [int(c) for c in best_coloring]
# EVOLVE-BLOCK-END
