"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 80.0
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Sort vertices by degree (descending) for better MRV
    degree = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: -degree[x])
    
    # For each pair of colors, track which edges connect them
    # We need to ensure that for any two colors c1, c2, the subgraph induced by vertices colored c1 or c2 is a forest.
    # This is equivalent to: for any two colors, the edges between them (and within each) form a forest.
    # Since adjacent vertices must have different colors, there are no edges within a single color class.
    # So the condition is: for any pair of colors (c1, c2), the edges between c1 and c2 form a forest.
    
    # We'll use a backtracking search with MRV and DSU for cycle detection.
    # State: color assignment for vertices, DSU for each color pair.
    
    # To make this efficient, we'll use a recursive backtracking with:
    # 1. MRV: pick the uncolored vertex with the fewest available colors.
    # 2. Forward checking: maintain for each uncolored vertex the set of available colors.
    # 3. DSU with rollback for each color pair to detect cycles.
    
    # However, implementing full DSU rollback for all color pairs is complex.
    # Alternative: for a given partial coloring, check if adding a vertex with a color creates a bicolored cycle.
    # A bicolored cycle involving the new vertex (u, color c) and existing colors would require a path between two neighbors of u with colors from {c, d} for some d, that doesn't use u.
    
    # Let's try a simpler approach first: try small k values (2, 3, 4, ...) and use backtracking with a global cycle check.
    
    best_result = None
    best_k = 50
    
    # Try k from 2 up to some limit
    for k in range(2, 10):
        if time.time() > deadline - 5:
            break
        
        # Try to find a valid k-coloring
        colors = [-1] * n
        # available[v] = set of colors available for vertex v
        available = [set(range(k)) for _ in range(n)]
        
        # For cycle detection, we'll maintain a DSU for each color pair.
        # But that's too much. Instead, we'll check cycles incrementally.
        
        # Let's use a different strategy: backtracking with a global check after each assignment.
        # But that's too slow.
        
        # Better: use the fact that for a fixed k, we can use a SAT-like approach or a more efficient backtracking.
        
        # Let's implement a backtracking that, when assigning color c to vertex u,
        # checks if it creates a cycle in the subgraph induced by any pair of colors.
        
        # To check this efficiently, we can maintain for each color pair (c1, c2) a DSU.
        # But with k up to 9, there are ~36 pairs, and 50 vertices, so it's feasible.
        
        # Actually, let's just do a simple backtracking with a global validity check at each step,
        # but prune early using forward checking.
        
        # For now, let's try a greedy + local search approach for small k.
        
        # Greedy coloring with MRV:
        assigned = [-1] * n
        for v in order:
            # Find the smallest color that doesn't conflict with neighbors
            used = set()
            for nb in adj[v]:
                if assigned[nb] != -1:
                    used.add(assigned[nb])
            for c in range(k):
                if c not in used:
                    assigned[v] = c
                    break
            else:
                # No color available, fail
                break
        
        if assigned[-1] != -1:
            # Check if valid: no bicolored cycles
            if _is_valid(assigned, adj, n, edges):
                best_result = assigned[:]
                best_k = k
                break
        
        # If greedy failed, try backtracking
        # (This is a placeholder; we'll implement a more thorough search if needed)
    
    if best_result is None:
        # Fallback: use a simple valid coloring with many colors
        best_result = list(range(n))
        best_k = n
    
    return best_result

def _is_valid(colors, adj, n, edges):
    # Check that adjacent vertices have different colors
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    
    # Check that for each pair of colors, the induced subgraph is a forest
    # Group vertices by color
    from collections import defaultdict
    color_to_vertices = defaultdict(list)
    for v in range(n):
        color_to_vertices[colors[v]].append(v)
    
    # For each pair of colors, check if the edges between them form a forest
    color_pairs_checked = set()
    for c1 in color_to_vertices:
        for c2 in color_to_vertices:
            if c1 >= c2:
                continue
            # Check edges between c1 and c2
            # Build DSU for vertices in c1 and c2
            vertices_in_pair = set(color_to_vertices[c1] + color_to_vertices[c2])
            parent = {v: v for v in vertices_in_pair}
            
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
            
            for u, v in edges:
                if colors[u] in (c1, c2) and colors[v] in (c1, c2):
                    if not union(u, v):
                        return False
    
    return True

# EVOLVE-BLOCK-END
