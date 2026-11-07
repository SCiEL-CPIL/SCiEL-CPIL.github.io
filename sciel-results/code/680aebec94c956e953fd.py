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

def _get_degeneracy_order(adj, n):
    """Get vertex ordering using degeneracy (peeling lowest-degree vertex iteratively)."""
    deg = [len(adj[i]) for i in range(n)]
    remaining = set(range(n))
    order = []
    deg_copy = deg[:]
    for _ in range(n):
        # Find vertex with minimum degree among remaining
        min_deg = float('inf')
        min_v = -1
        for v in remaining:
            if deg_copy[v] < min_deg:
                min_deg = deg_copy[v]
                min_v = v
        order.append(min_v)
        remaining.discard(min_v)
        for nb in adj[min_v]:
            if nb in remaining:
                deg_copy[nb] -= 1
    return order

def _find_coloring_csp(max_colors, deadline):
    """CSP-based search with degeneracy ordering and incremental bicolored-cycle check."""
    n = P['n']
    adj = _build_adj()
    order = _get_degeneracy_order(adj, n)
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i
    
    coloring = [-1] * n
    edges = P['edges']
    
    def check_bicolored_fast(vertex, color):
        """Fast check: only examine edges where both endpoints have colors in {color, c2}."""
        # Collect used colors
        used_colors = set()
        for v in range(n):
            if coloring[v] != -1:
                used_colors.add(coloring[v])
        
        for c2 in used_colors:
            if c2 == color:
                continue
            
            # Build subgraph of vertices with color in {color, c2}
            # and check for cycles using union-find on just those edges
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
                    return True
                parent[rx] = ry
                return False
            
            for u, v in edges:
                cu = coloring[u] if u != vertex else color
                cv = coloring[v] if v != vertex else color
                if (cu == color or cu == c2) and (cv == color or cv == c2):
                    if union(u, v):
                        return True
        return False
    
    # Track colors used for quick access
    colors_used = set()
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        
        v = order[idx]
        
        # Try colors in order, with symmetry breaking
        # First vertex in order must be 0
        # Second vertex in order must be 0 or 1
        # Third vertex in order must be 0, 1, or 2
        # This helps break symmetry
        
        if idx == 0:
            color_range = [0]
        elif idx == 1:
            color_range = [0, 1]
        elif idx == 2:
            color_range = [0, 1, 2]
        else:
            color_range = range(max_colors)
        
        for color in color_range:
            # Check proper coloring
            valid = True
            for nb in adj[v]:
                if coloring[nb] == color:
                    valid = False
                    break
            if not valid:
                continue
            
            coloring[v] = color
            colors_used.add(color)
            
            # Check bicolored cycle incrementally
            if not check_bicolored_fast(v, color):
                if backtrack(idx + 1):
                    return True
            
            coloring[v] = -1
            colors_used.discard(color)
        
        return False
    
    if backtrack(0):
        return coloring
    return None

def construct():
    deadline = time.time() + 85
    
    # Try increasing numbers of colors starting from 3
    # 2 colors is impossible as the graph has cycles
    for num_colors in range(3, 15):
        if time.time() > deadline - 3:
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
