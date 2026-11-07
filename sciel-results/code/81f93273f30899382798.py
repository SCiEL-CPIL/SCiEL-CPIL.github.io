import time
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_valid(coloring):
    """Check if coloring is valid: adjacent differ, and no bicolored cycles."""
    n = P['n']
    edges = P['edges']
    
    # Check adjacent vertices have different colors
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    
    colors_used = set(coloring)
    
    for c1 in colors_used:
        for c2 in colors_used:
            if c1 >= c2:
                continue
            
            parent = list(range(n))
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return True
                parent[rx] = ry
                return False
            
            has_cycle = False
            for u, v in edges:
                cu, cv = coloring[u], coloring[v]
                if (cu == c1 or cu == c2) and (cv == c1 or cv == c2):
                    if union(u, v):
                        has_cycle = True
                        break
            
            if has_cycle:
                return False
    
    return True

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_valid_coloring(k, deadline):
    """Find a valid k-coloring using backtracking with pruning."""
    n = P['n']
    edges = P['edges']
    adj = _build_adj()
    
    # Order vertices by degree descending for better pruning
    order = sorted(range(n), key=lambda v: -len(adj[v]))
    pos_in_order = {v: i for i, v in enumerate(order)}
    
    coloring = [-1] * n
    # For each color pair, track edges count and vertices count for cycle detection
    # Use union-find per color pair incrementally - too complex
    # Instead, do a lightweight check: count edges in each color pair subgraph
    
    def count_edges_in_pair(c1, c2):
        """Count edges where both endpoints are colored c1 or c2."""
        cnt = 0
        for u, v in edges:
            cu, cv = coloring[u], coloring[v]
            if cu >= 0 and cv >= 0:
                if (cu == c1 or cu == c2) and (cv == c1 or cv == c2):
                    cnt += 1
        return cnt
    
    def count_vertices_in_pair(c1, c2):
        """Count vertices colored c1 or c2."""
        cnt = 0
        for v in range(n):
            if coloring[v] == c1 or coloring[v] == c2:
                cnt += 1
        return cnt
    
    def has_bicolored_cycle(c1, c2):
        """Check if the subgraph induced by colors c1 and c2 has a cycle."""
        parent = list(range(n))
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return True
            parent[rx] = ry
            return False
        
        for u, v in edges:
            cu, cv = coloring[u], coloring[v]
            if (cu == c1 or cu == c2) and (cv == c1 or cv == c2):
                if union(u, v):
                    return True
        return False
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == n:
            # Verify full validity
            for u, v in edges:
                if coloring[u] == coloring[v]:
                    return False
            colors_used = set(coloring)
            for c1 in colors_used:
                for c2 in colors_used:
                    if c1 >= c2:
                        continue
                    if has_bicolored_cycle(c1, c2):
                        return False
            return True
        
        v = order[pos]
        
        for c in range(k):
            # Check adjacency
            valid = True
            for u in adj[v]:
                if coloring[u] == c:
                    valid = False
                    break
            
            if not valid:
                continue
            
            coloring[v] = c
            
            # Prune: check if any color pair involving c now has too many edges
            # A forest on m vertices has at most m-1 edges per component
            # Total edges <= total vertices - components <= total vertices - 1
            # So if edges > vertices - 1, there must be a cycle
            prune = False
            for c2 in range(k):
                if c2 == c:
                    continue
                # Check pair (c, c2)
                e = count_edges_in_pair(c, c2)
                vcnt = count_vertices_in_pair(c, c2)
                if vcnt > 0 and e > vcnt - 1:
                    prune = True
                    break
            
            if not prune:
                if backtrack(pos + 1):
                    return True
            
            coloring[v] = -1
        
        return False
    
    if backtrack(0):
        return [int(c) for c in coloring]
    return None

def construct():
    deadline = time.time() + 75  # 75s safety margin
    
    # Try 2 colors
    result = _find_valid_coloring(2, deadline)
    if result is not None:
        if _check_valid(result):
            return result
    
    # Try 3 colors
    result = _find_valid_coloring(3, deadline)
    if result is not None:
        if _check_valid(result):
            return result
    
    # Fallback: known 3-coloring
    n = P['n']
    coloring = [i % 3 for i in range(n)]
    if _check_valid(coloring):
        return [int(c) for c in coloring]
    
    # Last resort
    return [int(i) for i in range(n)]
