"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_bicolored_cycles(coloring, n, edges):
    """Check if any pair of colors induces a subgraph with a cycle using union-find."""
    color_to_vertices = defaultdict(list)
    for v in range(n):
        color_to_vertices[coloring[v]].append(v)
    
    colors_used = sorted(color_to_vertices.keys())
    
    for i in range(len(colors_used)):
        for j in range(i + 1, len(colors_used)):
            ci, cj = colors_used[i], colors_used[j]
            vertices_in_pair = set(color_to_vertices[ci]) | set(color_to_vertices[cj])
            
            # Union-find
            parent = {v: v for v in vertices_in_pair}
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                px, py = find(x), find(y)
                if px == py:
                    return False
                parent[px] = py
                return True
            
            for u, v in edges:
                if u in vertices_in_pair and v in vertices_in_pair:
                    if coloring[u] in (ci, cj) and coloring[v] in (ci, cj):
                        if not union(u, v):
                            return False
    
    return True

def _is_valid(coloring, n, edges):
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    return _check_bicolored_cycles(coloring, n, edges)

def _count_colors(coloring):
    return len(set(coloring))

def _try_modular_patterns(n, edges, k, deadline):
    """Try modular colorings: color[i] = (a*i + b) % k"""
    best = None
    best_count = n + 1
    for a in range(1, k):
        for b in range(k):
            if time.time() > deadline:
                return best, best_count
            coloring = [(i * a + b) % k for i in range(n)]
            if _is_valid(coloring, n, edges):
                cnt = _count_colors(coloring)
                if cnt < best_count:
                    best_count = cnt
                    best = coloring[:]
                    if cnt == 3:
                        return best, best_count
    return best, best_count

def _backtracking_search(n, edges, k, deadline):
    """Backtracking search for k-coloring with bicolored cycle constraint."""
    colors = [-1] * n
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Order vertices by degree (all degree 4 here, so use natural order)
    # But we can use a smarter ordering: start with vertex 0
    
    def backtrack(v):
        if time.time() > deadline:
            return False
        if v == n:
            # Final check for bicolored cycles
            return _check_bicolored_cycles(colors, n, edges)
        
        # Try each color
        for c in range(k):
            # Check adjacency constraint
            valid = True
            for nb in adj[v]:
                if colors[nb] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            colors[v] = c
            
            # Forward checking: for each uncolored neighbor, check if it still has options
            # Also check if this creates a bicolored cycle with any pair
            # Quick check: for each pair of colors used so far that includes c,
            # check if adding v creates a cycle
            feasible = True
            for nb in adj[v]:
                if colors[nb] == -1:
                    # Check if nb still has at least one available color
                    available = set(range(k))
                    for nnb in adj[nb]:
                        if colors[nnb] != -1:
                            available.discard(colors[nnb])
                    if not available:
                        feasible = False
                        break
            
            if feasible:
                if backtrack(v + 1):
                    return True
            
            colors[v] = -1
        
        return False
    
    if backtrack(0):
        return colors[:]
    return None

def construct():
    n = P['n']
    edges = P['edges']
    
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit, 10s safety margin
    
    best_coloring = list(range(n))  # Fallback: 50 colors
    best_count = n
    
    # Try k from 3 upward
    for k in range(3, n + 1):
        if time.time() > deadline:
            break
        
        # First try modular patterns (fast)
        mod_result, mod_count = _try_modular_patterns(n, edges, k, deadline)
        if mod_result is not None and mod_count < best_count:
            best_count = mod_count
            best_coloring = mod_result
            if best_count == 3:
                break
        
        # If modular didn't find k=3, try backtracking
        if k == 3 and best_count > 3:
            bt_result = _backtracking_search(n, edges, 3, deadline)
            if bt_result is not None:
                best_count = 3
                best_coloring = bt_result
                break
        
        # For k >= 4, also try backtracking with limited time
        if k <= 5 and time.time() < deadline - 20:
            bt_result = _backtracking_search(n, edges, k, deadline)
            if bt_result is not None:
                cnt = _count_colors(bt_result)
                if cnt < best_count:
                    best_count = cnt
                    best_coloring = bt_result
                    if best_count == k:
                        break
        
        if best_count <= k:
            break
    
    # Final verification
    if not _is_valid(best_coloring, n, edges):
        best_coloring = list(range(n))
    
    result = [int(c) for c in best_coloring]
    return result
# EVOLVE-BLOCK-END
