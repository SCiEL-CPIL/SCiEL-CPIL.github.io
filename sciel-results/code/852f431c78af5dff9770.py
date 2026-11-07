import time
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_valid(coloring):
    """Check if coloring is valid: adjacent differ, and no bicolored cycles."""
    n = P['n']
    edges = P['edges']
    
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
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _try_structural_coloring(max_colors):
    """
    Try a greedy structural coloring based on the ladder-like structure.
    The graph appears to be a 3-rung ladder: columns of 3 vertices connected
    in a path, with rungs between columns.
    """
    n = P['n']
    adj = _build_adj()
    edges = P['edges']
    
    # Try several systematic colorings based on the graph's column structure
    # Columns appear to be: col[i] = {3*i, 3*i+1, 3*i+2} for i in 0..16
    # With wrap-around edges connecting col[0] and col[16]
    
    # Strategy: use a repeating pattern that avoids bicolored cycles
    # For a ladder graph, 3 colors can work if we use a proper 3-coloring
    # of the "skeleton" (the cycle of column-centers)
    
    # Try: color vertex v based on its position in the column structure
    # col = v // 3, row = v % 3
    # Use pattern: color = (col + row) % max_colors
    
    # But we need to verify no bicolored cycles exist
    
    patterns = []
    
    # Pattern 1: (col + row) % k
    for k in range(3, max_colors + 1):
        coloring = []
        for v in range(n):
            col = v // 3
            row = v % 3
            coloring.append((col + row) % k)
        patterns.append(coloring)
    
    # Pattern 2: (col * 2 + row) % k
    for k in range(3, max_colors + 1):
        coloring = []
        for v in range(n):
            col = v // 3
            row = v % 3
            coloring.append((col * 2 + row) % k)
        patterns.append(coloring)
    
    # Pattern 3: (col + 2*row) % k
    for k in range(3, max_colors + 1):
        coloring = []
        for v in range(n):
            col = v // 3
            row = v % 3
            coloring.append((col + 2 * row) % k)
        patterns.append(coloring)
    
    # Pattern 4: row-based (color by row within column)
    for k in range(3, max_colors + 1):
        coloring = []
        for v in range(n):
            col = v // 3
            row = v % 3
            coloring.append(row % k)
        patterns.append(coloring)
    
    # Pattern 5: (col % k)
    for k in range(3, max_colors + 1):
        coloring = []
        for v in range(n):
            col = v // 3
            row = v % 3
            coloring.append(col % k)
        patterns.append(coloring)
    
    # Pattern 6: (col % k + (row % 2)) % k
    for k in range(3, max_colors + 1):
        coloring = []
        for v in range(n):
            col = v // 3
            row = v % 3
            coloring.append((col % k + (row % 2)) % k)
        patterns.append(coloring)
    
    # Pattern 7: (col % k) XOR (row % 2)
    for k in range(3, max_colors + 1):
        coloring = []
        for v in range(n):
            col = v // 3
            row = v % 3
            coloring.append((col % k) ^ (row % 2))
        patterns.append(coloring)
    
    # Pattern 8: (col + row * 3) % k
    for k in range(3, max_colors + 1):
        coloring = []
        for v in range(n):
            col = v // 3
            row = v % 3
            coloring.append((col + row * 3) % k)
        patterns.append(coloring)
    
    # Pattern 9: (col * 3 + row) % k
    for k in range(3, max_colors + 1):
        coloring = []
        for v in range(n):
            col = v // 3
            row = v % 3
            coloring.append((col * 3 + row) % k)
        patterns.append(coloring)
    
    # Pattern 10: (col % k) + (row == 1)  # offset middle row
    for k in range(3, max_colors + 1):
        coloring = []
        for v in range(n):
            col = v // 3
            row = v % 3
            c = col % k
            if row == 1:
                c = (c + 1) % k
            elif row == 2:
                c = (c + 2) % k
            coloring.append(c)
        patterns.append(coloring)
    
    # Try each pattern
    for coloring in patterns:
        if _check_valid(coloring):
            # Count distinct colors used
            used = set(coloring)
            if len(used) <= max_colors:
                return coloring
    
    return None

def _find_coloring_csp(max_colors, deadline):
    """CSP-based search with MRV heuristic and fast incremental bicolored-cycle check."""
    n = P['n']
    adj = _build_adj()
    edges = P['edges']
    
    # Precompute edge list for fast access
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
    
    coloring = [-1] * n
    
    # Fast bicolored cycle check: only check pairs involving the newly assigned vertex
    def check_bicolored_fast(vertex, color):
        """
        Check if assigning 'color' to 'vertex' creates a bicolored cycle.
        Only need to check color pairs (color, c2) where c2 is a color already in use.
        For each such pair, check if the subgraph induced by vertices with those two colors has a cycle.
        """
        # Collect used colors
        used_colors = set()
        for v in range(n):
            if coloring[v] != -1:
                used_colors.add(coloring[v])
        
        for c2 in used_colors:
            if c2 == color:
                continue
            
            # Build subgraph of vertices colored 'color' or 'c2'
            # Use union-find on just these vertices
            # A bicolored cycle exists iff some edge in this subgraph connects two vertices already in the same component
            
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
                    return True  # cycle detected
                parent[rx] = ry
                return False
            
            has_cycle = False
            for u, v in edges:
                cu = coloring[u] if u != vertex else color
                cv = coloring[v] if v != vertex else color
                if (cu == color or cu == c2) and (cv == color or cv == c2):
                    if union(u, v):
                        has_cycle = True
                        break
            
            if has_cycle:
                return True
        
        return False
    
    # Get initial ordering by degree (ascending)
    deg = [len(adj[i]) for i in range(n)]
    
    # MRV: dynamically choose vertex with fewest available colors
    def get_mrv_vertex():
        best_v = -1
        best_count = n + 1
        for v in range(n):
            if coloring[v] != -1:
                continue
            # Count available colors
            neighbor_colors = set()
            for nb in adj[v]:
                if coloring[nb] != -1:
                    neighbor_colors.add(coloring[nb])
            avail = max_colors - len(neighbor_colors)
            if avail < best_count:
                best_count = avail
                best_v = v
                if avail <= 1:
                    break
        return best_v
    
    def backtrack():
        if time.time() > deadline:
            return False
        
        v = get_mrv_vertex()
        if v == -1:
            return True  # all vertices colored
        
        # Try colors, prefer those not used by neighbors
        neighbor_colors = set()
        for nb in adj[v]:
            if coloring[nb] != -1:
                neighbor_colors.add(coloring[nb])
        
        # Sort colors: prefer unused ones first
        color_order = list(range(max_colors))
        color_order.sort(key=lambda c: (c in neighbor_colors, c))
        
        for color in color_order:
            if color in neighbor_colors:
                continue
            
            coloring[v] = color
            
            if not check_bicolored_fast(v, color):
                if backtrack():
                    return True
            
            coloring[v] = -1
        
        return False
    
    if backtrack():
        return coloring
    return None

def construct():
    deadline = time.time() + 85
    
    # First try structural colorings (very fast)
    for num_colors in range(3, 15):
        if time.time() > deadline - 10:
            break
        coloring = _try_structural_coloring(num_colors)
        if coloring is not None:
            if _check_valid(coloring):
                return [int(c) for c in coloring]
    
    # Then try CSP search
    for num_colors in range(3, 15):
        if time.time() > deadline - 5:
            break
        coloring = _find_coloring_csp(num_colors, deadline)
        if coloring is not None:
            if _check_valid(coloring):
                return [int(c) for c in coloring]
    
    # Fallback: identity coloring
    return [int(i) for i in range(P['n'])]

if __name__ == '__main__':
    result = construct()
    print(result)
    print(len(set(result)))
