"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency():
    adj = defaultdict(list)
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_bicolor_forest(colors, adj):
    """Check that for every pair of colors, the induced subgraph is a forest."""
    n = P['n']
    # Group vertices by color
    color_groups = defaultdict(list)
    for i in range(n):
        color_groups[colors[i]].append(i)
    
    colors_used = sorted(color_groups.keys())
    
    for idx_c1 in range(len(colors_used)):
        for idx_c2 in range(idx_c1 + 1, len(colors_used)):
            c1, c2 = colors_used[idx_c1], colors_used[idx_c2]
            # Build induced subgraph on vertices with color c1 or c2
            vertices = set(color_groups[c1]) | set(color_groups[c2])
            if not vertices:
                continue
            
            # Union-Find to detect cycles
            parent = {v: v for v in vertices}
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False  # Cycle detected
                parent[rx] = ry
                return True
            
            # Check all edges in the induced subgraph
            for u in vertices:
                for v in adj[u]:
                    if v in vertices and u < v:
                        if colors[u] in (c1, c2) and colors[v] in (c1, c2):
                            if not union(u, v):
                                return False  # Cycle found
    return True

def _count_distinct_colors(colors):
    return len(set(colors))

def _try_pattern_and_optimize(pattern_func, max_colors, adj):
    """Try a coloring pattern, then use local search to reduce colors if possible."""
    n = P['n']
    colors = [pattern_func(i) for i in range(n)]
    
    # First, verify proper coloring and bicolored forest constraint
    valid = True
    for u, v in P['edges']:
        if colors[u] == colors[v]:
            valid = False
            break
    if not valid:
        return None
    
    if not _check_bicolor_forest(colors, adj):
        return None
    
    # Try to reduce colors by merging
    current_colors = colors[:]
    distinct = _count_distinct_colors(current_colors)
    
    # Local search: try to reassign vertices to reduce color count
    for _ in range(1000):
        improved = False
        for i in range(n):
            old_color = current_colors[i]
            # Try assigning a different existing color
            for new_color in range(max_colors):
                if new_color == old_color:
                    continue
                # Check proper coloring
                proper = True
                for v in adj[i]:
                    if current_colors[v] == new_color:
                        proper = False
                        break
                if not proper:
                    continue
                
                # Tentatively assign and check bicolored forest
                current_colors[i] = new_color
                if _check_bicolor_forest(current_colors, adj):
                    distinct_new = _count_distinct_colors(current_colors)
                    if distinct_new < distinct:
                        distinct = distinct_new
                        improved = True
                        break
                    else:
                        # Keep if same distinct count but might help future moves
                        current_colors[i] = old_color
                else:
                    current_colors[i] = old_color
        
        if not improved:
            break
    
    return current_colors

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    adj = _build_adjacency()
    n = P['n']
    
    # Try various patterns
    best_result = None
    best_score = n + 1
    
    # Pattern 1: Simple modular 4-coloring
    def pattern_mod4(i):
        return i % 4
    
    # Pattern 2: Based on grid structure (5x10)
    def pattern_grid(i):
        row = i // 10
        col = i % 10
        return (row + col) % 4
    
    # Pattern 3: Different grid offset
    def pattern_grid2(i):
        row = i // 10
        col = i % 10
        return (row * 2 + col) % 4
    
    # Pattern 4: Checkerboard-like
    def pattern_checker(i):
        row = i // 10
        col = i % 10
        return (row + col) % 2
    
    # Pattern 5: 3-coloring attempt
    def pattern_mod3(i):
        return i % 3
    
    patterns = [pattern_mod4, pattern_grid, pattern_grid2, pattern_checker, pattern_mod3]
    
    for pat in patterns:
        if time.time() > deadline:
            break
        
        result = _try_pattern_and_optimize(pat, 10, adj)
        if result is not None:
            score = _count_distinct_colors(result)
            if score < best_score:
                best_score = score
                best_result = result[:]
        
        # If we found a 2-coloring, we're done (optimal for non-bipartite)
        if best_score <= 2:
            break
    
    # If no pattern worked, fall back to greedy
    if best_result is None:
        # Greedy coloring
        colors = [0] * n
        for i in range(n):
            used = set()
            for v in adj[i]:
                if v < i:
                    used.add(colors[v])
            for c in range(10):
                if c not in used:
                    colors[i] = c
                    break
            if colors[i] == 0 and 0 in used:
                colors[i] = max(used) + 1 if max(used) + 1 < 10 else 0
        
        # Verify and fix if needed
        if _check_bicolor_forest(colors, adj):
            best_result = colors
            best_score = _count_distinct_colors(colors)
        else:
            # Last resort: unique colors
            best_result = list(range(n))
            best_score = n
    
    return [int(c) for c in best_result]
# EVOLVE-BLOCK-END
