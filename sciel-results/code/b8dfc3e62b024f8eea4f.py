"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_bicolored_forest(colors, n, edges):
    """Check that for every pair of colors, the induced subgraph is a forest."""
    if len(colors) != n:
        return False
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
        
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = colors[u], colors[v]
        key = (min(cu, cv), max(cu, cv))
        pair_edges[key].append((u, v))
    
    for pair, elist in pair_edges.items():
        parent = list(range(n))
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
        for u, v in elist:
            if not union(u, v):
                return False
    return True

def _solve_backtracking(n, edges, num_colors, deadline, rng):
    """
    Advanced backtracking with Forward Checking and MCV/LCV heuristics.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Initial domains
    domains = [set(range(num_colors)) for _ in range(n)]
    colors = [-1] * n
    
    # Precompute edge list for quick access in forest check
    # We will maintain the forest property incrementally or check locally.
    # For simplicity and robustness, we will use a local check function that is fast.
    
    def get_neighbors_colors(v, colors):
        return [colors[u] for u in adj[v] if colors[u] != -1]

    def is_valid_color(v, c, colors, domains):
        # Check neighbor colors
        for u in adj[v]:
            if colors[u] == c:
                return False
        return True

    def check_forest_constraint(v, c, colors):
        """
        Check if assigning color c to v creates a cycle in any bicolored subgraph.
        Only need to check pairs (c, colors[u]) for all neighbors u.
        """
        for u in adj[v]:
            cu = colors[u]
            if cu == -1 or cu == c:
                continue
            c1, c2 = min(c, cu), max(c, cu)
            
            # Collect edges for this pair
            # Optimization: We can maintain a global structure, but for N=50, 
            # scanning edges is fast enough if done rarely.
            # However, in backtracking, we do this often.
            # Let's just scan the edges involved in this pair.
            
            # To speed up, we can pre-filter edges by color pair if we maintained them.
            # Here we just scan all edges and filter.
            pair_edges = []
            for e_u, e_v in edges:
                cu_e, cv_e = colors[e_u], colors[e_v]
                if e_u == v: cu_e = c
                if e_v == v: cv_e = c
                if cu_e == -1 or cv_e == -1 or cu_e == cv_e:
                    continue
                if (min(cu_e, cv_e), max(cu_e, cv_e)) == (c1, c2):
                    pair_edges.append((e_u, e_v))
            
            # Union-Find
            parent = list(range(n))
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
            
            for u_e, v_e in pair_edges:
                if not union(u_e, v_e):
                    return False
        return True

    def backtrack():
        if time.time() > deadline:
            return False
        
        # MCV: Select unassigned vertex with smallest domain
        unassigned = [i for i in range(n) if colors[i] == -1]
        if not unassigned:
            return True
        
        # Sort by domain size
        unassigned.sort(key=lambda x: len(domains[x]))
        
        v = unassigned[0]
        
        # LCV: Order colors by how many options they leave for neighbors
        # Approximation: Prefer colors that are less frequent in neighbors
        neighbor_colors = [colors[u] for u in adj[v] if colors[u] != -1]
        
        def color_score(c):
            # Count how many neighbors already have this color (should be 0 if valid)
            # Count how many neighbors would have this color reduced?
            # Simple: Random order or sorted order
            return 0
            
        # Just try colors in domain
        for c in list(domains[v]):
            if not is_valid_color(v, c, colors, domains):
                continue
                
            if not check_forest_constraint(v, c, colors):
                continue
            
            # Assign
            colors[v] = c
            old_domains = [set(d) for d in domains]
            
            # Forward Checking: Update domains of neighbors
            valid = True
            for u in adj[v]:
                if colors[u] == -1:
                    if c in domains[u]:
                        domains[u].discard(c)
                        if not domains[u]:
                            valid = False
                            break
            
            if valid:
                if backtrack():
                    return True
            
            # Backtrack
            colors[v] = -1
            domains = old_domains
            
        return False

    if backtrack():
        return colors
    return None

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 85
    
    rng = random.Random(42)
    best = None
    best_num_colors = n
    
    # Try from 3 colors upward
    for num_colors in range(3, n + 1):
        if best_num_colors <= num_colors:
            break
        if time.time() > deadline:
            break
        
        result = _solve_backtracking(n, edges, num_colors, deadline, rng)
        if result is not None:
            # Final verification
            if _check_bicolored_forest(result, n, edges):
                best = result
                best_num_colors = num_colors
                break
    
    if best is None:
        best = list(range(n))
    
    return [int(c) for c in best]
# EVOLVE-BLOCK-END
