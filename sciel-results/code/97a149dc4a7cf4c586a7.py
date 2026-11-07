import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adj():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

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

def _get_degeneracy_order(adj, n):
    """Get vertex ordering using degeneracy (peeling lowest-degree vertex iteratively)."""
    deg_copy = [len(adj[i]) for i in range(n)]
    remaining = list(range(n))
    order = []
    for _ in range(n):
        min_idx = 0
        min_deg = deg_copy[remaining[0]]
        for i in range(1, len(remaining)):
            if deg_copy[remaining[i]] < min_deg:
                min_deg = deg_copy[remaining[i]]
                min_idx = i
        v = remaining[min_idx]
        order.append(v)
        remaining.pop(min_idx)
        for nb in adj[v]:
            if nb in remaining:
                deg_copy[nb] -= 1
    return order

def _find_coloring_csp(max_colors, deadline, adj, order):
    """CSP-based search with degeneracy ordering and incremental bicolored-cycle check."""
    n = P['n']
    edges = P['edges']
    
    # Precompute for each vertex, which edges are incident
    vertex_edges = defaultdict(list)
    for i, (u, v) in enumerate(edges):
        vertex_edges[u].append(i)
        vertex_edges[v].append(i)
    
    coloring = [-1] * n
    
    # For fast bicolored cycle check, maintain for each color pair the edges in that subgraph
    # and use union-find incrementally
    
    def check_bicolored_with_new_vertex(vertex, color, coloring):
        """Check if adding vertex with given color creates a bicolored cycle."""
        # Get all colors currently used
        used_colors = set()
        for v in range(n):
            c = coloring[v]
            if c != -1 and c != color:
                used_colors.add(c)
        
        for c2 in used_colors:
            # Check subgraph induced by colors {color, c2}
            # Only need to check if the new vertex creates a cycle
            # Find all vertices with color in {color, c2}
            # and check connectivity using union-find on relevant edges
            
            # Quick check: find neighbors of vertex with color c2
            # If vertex has two neighbors with color c2 that are already connected
            # in the {color, c2} subgraph, we have a cycle
            
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
            
            # Build the subgraph
            for u, v in edges:
                cu = coloring[u] if coloring[u] != -1 else None
                cv = coloring[v] if coloring[v] != -1 else None
                
                # Include vertex with its assigned color
                if u == vertex:
                    cu = color
                if v == vertex:
                    cv = color
                
                if cu is None or cv is None:
                    continue
                if (cu == color or cu == c2) and (cv == color or cv == c2):
                    if union(u, v):
                        return True
        return False
    
    colors_used = set()
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        
        v = order[idx]
        
        # Try colors
        if idx == 0:
            color_range = [0]
        elif idx == 1:
            color_range = [0, 1]
        elif idx == 2:
            color_range = [0, 1, 2]
        else:
            color_range = list(range(max_colors))
        
        for color in color_range:
            # Check proper coloring: no neighbor has same color
            valid = True
            for nb in adj[v]:
                if coloring[nb] == color:
                    valid = False
                    break
            if not valid:
                continue
            
            coloring[v] = color
            colors_used.add(color)
            
            # Check bicolored cycle
            if not check_bicolored_with_new_vertex(v, color, coloring):
                if backtrack(idx + 1):
                    return True
            
            coloring[v] = -1
            colors_used.discard(color)
        
        return False
    
    if backtrack(0):
        return coloring
    return None

def construct():
    deadline = time.time() + 80
    
    adj = _build_adj()
    n = P['n']
    order = _get_degeneracy_order(adj, n)
    
    # Try increasing numbers of colors
    for num_colors in range(3, 20):
        if time.time() > deadline - 5:
            break
        
        coloring = _find_coloring_csp(num_colors, deadline, adj, order)
        if coloring is not None:
            if _check_valid(coloring):
                return [int(c) for c in coloring]
    
    # Fallback: use a greedy coloring that avoids bicolored cycles
    # Assign colors in a pattern that breaks cycles
    coloring = list(range(n))
    return [int(c) for c in coloring]

if __name__ == '__main__':
    result = construct()
    print(result)
    print(len(set(result)))
