"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency():
    adj = defaultdict(list)
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_bicolor_forest(colors, adj, n):
    """Check that for every pair of colors, the induced subgraph is a forest."""
    color_groups = defaultdict(list)
    for i in range(n):
        color_groups[colors[i]].append(i)
    
    colors_used = list(color_groups.keys())
    
    for i in range(len(colors_used)):
        for j in range(i + 1, len(colors_used)):
            c1, c2 = colors_used[i], colors_used[j]
            vertices = set(color_groups[c1]) | set(color_groups[c2])
            if len(vertices) <= 1:
                continue
            
            parent = {v: v for v in vertices}
            
            def find(x):
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
            
            for u in vertices:
                for v in adj[u]:
                    if v in vertices and u < v:
                        if not union(u, v):
                            return False
    return True

def _is_valid(colors, adj, n):
    for u, v in P['edges']:
        if colors[u] == colors[v]:
            return False
    return _check_bicolor_forest(colors, adj, n)

def _count_distinct(colors):
    return len(set(colors))

def _try_pattern(colors, adj, n, deadline):
    """Try a pattern, validate, then local search to reduce colors."""
    if time.time() > deadline:
        return None, 999
    
    if not _is_valid(colors, adj, n):
        return None, 999
    
    current = colors[:]
    distinct = _count_distinct(current)
    
    # Local search: try to reassign vertices to reduce distinct color count
    for iteration in range(500):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            old_c = current[i]
            # Try to change to an existing color or a lower unused color
            candidate_colors = set(current)
            # Also try colors 0..max_color to potentially reduce range
            for c in range(max(candidate_colors) + 1):
                if c == old_c:
                    continue
                # Check proper coloring
                proper = True
                for v in adj[i]:
                    if current[v] == c:
                        proper = False
                        break
                if not proper:
                    continue
                
                current[i] = c
                if _check_bicolor_forest(current, adj, n):
                    new_distinct = _count_distinct(current)
                    if new_distinct < distinct:
                        distinct = new_distinct
                        improved = True
                        break
                    else:
                        current[i] = old_c
                else:
                    current[i] = old_c
            if improved:
                break
        if not improved:
            break
    
    return current, distinct

def construct():
    start_time = time.time()
    deadline = start_time + 80
    n = P['n']
    adj = _build_adjacency()
    
    best_result = None
    best_score = n + 1
    
    # Generate various structured patterns
    patterns = []
    
    # Pattern 1: Row-based (5 rows of 10)
    for row_color in range(5):
        for col_offset in range(10):
            colors = [(i // 10 + row_color) % 5 for i in range(n)]
            patterns.append(colors)
    
    # Pattern 2: Column-based
    for col_color in range(10):
        for row_offset in range(5):
            colors = [(i % 10 + col_color) % 10 for i in range(n)]
            patterns.append(colors)
    
    # Pattern 3: Diagonal
    for offset in range(15):
        colors = [(i // 10 + i % 10 + offset) % 5 for i in range(n)]
        patterns.append(colors)
    
    # Pattern 4: Checkerboard-like
    for phase in range(4):
        colors = [((i // 10) % 2 + (i % 10) % 2 + phase) % 4 for i in range(n)]
        patterns.append(colors)
    
    # Pattern 5: Modular with period 3, 4, 6, 7
    for period in [3, 4, 6, 7]:
        for offset in range(period):
            colors = [(i + offset) % period for i in range(n)]
            patterns.append(colors)
    
    # Pattern 6: Two-dimensional modular
    for mod_r in [3, 4, 5]:
        for mod_c in [3, 4, 5]:
            for offset_r in range(mod_r):
                for offset_c in range(mod_c):
                    colors = [((i // 10 + offset_r) % mod_r) * 10 + ((i % 10 + offset_c) % mod_c) for i in range(n)]
                    # Remap to consecutive 0..max
                    unique = sorted(set(colors))
                    mapping = {v: idx for idx, v in enumerate(unique)}
                    colors = [mapping[c] for c in colors]
                    patterns.append(colors)
    
    # Random patterns
    for seed in range(200):
        if time.time() > deadline:
            break
        random.seed(seed)
        colors = [random.randint(0, 4) for _ in range(n)]
        patterns.append(colors)
    
    # Try each pattern
    for colors in patterns:
        if time.time() > deadline:
            break
        result, score = _try_pattern(colors, adj, n, deadline)
        if result is not None and score < best_score:
            best_score = score
            best_result = result[:]
            if best_score <= 3:
                break
    
    # If nothing good found, try more aggressive random search
    if best_score > 6:
        for seed in range(500):
            if time.time() > deadline:
                break
            random.seed(seed + 10000)
            colors = [random.randint(0, 7) for _ in range(n)]
            result, score = _try_pattern(colors, adj, n, deadline)
            if result is not None and score < best_score:
                best_score = score
                best_result = result[:]
                if best_score <= 4:
                    break
    
    # Fallback: trivial unique coloring
    if best_result is None:
        best_result = list(range(n))
        best_score = n
    
    return [int(c) for c in best_result]
# EVOLVE-BLOCK-END
