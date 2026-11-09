"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_bicolor_forest(colors, adj, n):
    """
    Check bicolored forest constraint using union-find.
    For each pair of colors, the induced subgraph must be a forest.
    """
    color_to_verts = {}
    for i in range(n):
        c = colors[i]
        if c not in color_to_verts:
            color_to_verts[c] = []
        color_to_verts[c].append(i)
    
    colors_used = sorted(color_to_verts.keys())
    nc = len(colors_used)
    
    for ci in range(nc):
        for cj in range(ci + 1, nc):
            c1 = colors_used[ci]
            c2 = colors_used[cj]
            verts = color_to_verts[c1] + color_to_verts[c2]
            vs = set(verts)
            if len(verts) < 2:
                continue
            
            parent = {v: v for v in verts}
            rank = {v: 0 for v in verts}
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True
            
            for u in verts:
                for v in adj[u]:
                    if v in vs and u < v:
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

def _local_search_reduce(colors, adj, n, deadline):
    """Local search to reduce number of distinct colors."""
    current = colors[:]
    distinct = _count_distinct(current)
    
    for _ in range(500):
        if time.time() > deadline:
            break
        improved = False
        for v in range(n):
            old_c = current[v]
            # Try all currently used colors and lower indices
            used = set(current)
            candidates = sorted(used)
            # Also try 0 to old_c-1 to encourage consolidation
            for tc in range(max(0, old_c - 2), old_c):
                if tc not in candidates:
                    candidates.append(tc)
            candidates = list(set(candidates))
            
            for tc in candidates:
                if tc == old_c:
                    continue
                # Proper coloring check
                proper = True
                for u in adj[v]:
                    if current[u] == tc:
                        proper = False
                        break
                if not proper:
                    continue
                current[v] = tc
                if _check_bicolor_forest(current, adj, n):
                    new_d = _count_distinct(current)
                    if new_d < distinct:
                        distinct = new_d
                        improved = True
                        break
                    else:
                        current[v] = old_c
                else:
                    current[v] = old_c
            if improved:
                break
        if not improved:
            break
    
    return current, distinct

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    n = P['n']
    adj = _build_adjacency()
    
    best_result = None
    best_score = n + 1
    
    # Grid structure: 5 rows x 10 columns, vertex i is at (i//10, i%10)
    # With wraparound: row i wraps to row (i+1)%5, col j wraps to col (j+1)%10
    
    # Generate diverse patterns exploiting the toroidal grid structure
    patterns = []
    
    # Pattern type 1: row-based with column offset
    for k in range(3, 15):
        for row_mult in range(0, k):
            for col_mult in range(0, k):
                if row_mult == 0 and col_mult == 0:
                    continue
                pat = [(r * row_mult + c * col_mult) % k for r in range(5) for c in range(10)]
                patterns.append(pat)
    
    # Pattern type 2: diagonal
    for k in range(3, 15):
        for shift in range(k):
            pat = [(r + c + shift) % k for r in range(5) for c in range(10)]
            patterns.append(pat)
    
    # Pattern type 3: row only
    for k in range(3, 15):
        pat = [r % k for r in range(5) for _ in range(10)]
        patterns.append(pat)
    
    # Pattern type 4: column only
    for k in range(3, 15):
        pat = [c % k for _ in range(5) for c in range(10)]
        patterns.append(pat)
    
    # Pattern type 5: 2D checkerboard variants
    for k in range(3, 15):
        for a in range(k):
            for b in range(k):
                pat = [(r * a + c * b) % k for r in range(5) for c in range(10)]
                patterns.append(pat)
    
    # Deduplicate patterns
    seen = set()
    unique_patterns = []
    for p in patterns:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique_patterns.append(p)
    
    # Sort by number of distinct colors (fewer first)
    unique_patterns.sort(key=lambda p: len(set(p)))
    
    # Try patterns
    for pat in unique_patterns:
        if time.time() > deadline - 5.0:
            break
        if len(set(pat)) >= best_score:
            continue
        if _is_valid(pat, adj, n):
            distinct = _count_distinct(pat)
            if distinct < best_score:
                # Try to reduce further with local search
                reduced, reduced_distinct = _local_search_reduce(pat, adj, n, deadline)
                if reduced_distinct < best_score:
                    best_score = reduced_distinct
                    best_result = reduced[:]
                else:
                    best_score = distinct
                    best_result = pat[:]
    
    # If no pattern worked well, try random restarts
    if best_score > 8:
        import random
        random.seed(42)
        for attempt in range(20):
            if time.time() > deadline - 5.0:
                break
            # Random pattern with k colors
            k = 3 + attempt % 5
            pat = [random.randint(0, k-1) for _ in range(n)]
            # Ensure proper coloring: fix conflicts
            for _ in range(100):
                ok = True
                for u, v in P['edges']:
                    if pat[u] == pat[v]:
                        pat[v] = random.randint(0, k-1)
                        ok = False
                if ok:
                    break
            if _is_valid(pat, adj, n):
                distinct = _count_distinct(pat)
                if distinct < best_score:
                    reduced, reduced_distinct = _local_search_reduce(pat, adj, n, deadline)
                    if reduced_distinct < best_score:
                        best_score = reduced_distinct
                        best_result = reduced[:]
    
    # Fallback
    if best_result is None:
        best_result = list(range(n))
        best_score = n
    
    return [int(c) for c in best_result]
# EVOLVE-BLOCK-END
