"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_bicolored_forest(colors, edges):
    """Check if for every pair of colors, the induced subgraph is a forest."""
    from collections import defaultdict
    n = len(colors)
    color_map = defaultdict(list)
    for v in range(n):
        color_map[colors[v]].append(v)
    
    colors_used = set(colors)
    for i, c1 in enumerate(colors_used):
        for c2 in colors_used:
            if c1 >= c2:
                continue
            # Build subgraph with only vertices of color c1 or c2
            verts = [v for v in range(n) if colors[v] in (c1, c2)]
            if len(verts) < 3:
                continue
            vert_set = set(verts)
            # Build adjacency in this subgraph
            adj = defaultdict(list)
            for u, v in edges:
                if u in vert_set and v in vert_set:
                    adj[u].append(v)
                    adj[v].append(u)
            # Check for cycles using union-find
            parent = {}
            def find(x):
                if x not in parent:
                    parent[x] = x
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
            for u in verts:
                find(u)
            for u, v in edges:
                if u in vert_set and v in vert_set:
                    if not union(u, v):
                        return False
    return True

def _validate(colors, edges):
    """Validate coloring: adjacent differ, no bicolored cycles."""
    n = len(colors)
    # Check adjacent vertices differ
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    # Check bicolored forests
    if not _check_bicolored_forest(colors, edges):
        return False
    return True

def construct():
    n = P['n']
    edges = P['edges']
    
    # Try small number of colors first, increasing
    # For a degree-4 graph, 3 colors might work if it's planar-like
    # But the bicolored forest constraint is stronger
    # Let's try 3, 4, 5 colors
    
    # First, try a greedy/backtracking approach for small color counts
    # Then fall back to MILP if needed
    
    # Try to find a valid coloring with k colors for k = 3, 4, 5, ...
    for k in range(3, 8):
        # Use backtracking search
        colors = [-1] * n
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Order vertices by degree (descending) for better pruning
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        pos_in_order = [0] * n
        for i, v in enumerate(order):
            pos_in_order[v] = i
        
        # Backtracking
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            for c in range(k):
                # Check if color c is valid for v
                valid = True
                for nb in adj[v]:
                    if colors[nb] == c:
                        valid = False
                        break
                if not valid:
                    continue
                colors[v] = c
                # Check bicolored forest constraint incrementally is expensive
                # For now, just do full check at the end
                if backtrack(idx + 1):
                    return True
                colors[v] = -1
            return False
        
        if backtrack(0):
            # Final validation
            if _validate(colors, edges):
                return [int(c) for c in colors]
        
        # If backtracking fails, try MILP
        # Formulate MILP for k colors
        # Variables: x[v][c] = 1 if vertex v has color c
        # For each pair of colors (c1, c2), we need to ensure no cycles
        # This is complex. Instead, use a simpler approach:
        # Enumerate all cycles in the graph and add constraints
        
        # Find all simple cycles of length 4 and 5 (small cycles are most problematic)
        # For each cycle, at least one edge must have its two endpoints with different colors
        # AND for bicolored constraint: for any pair of colors, the induced subgraph is acyclic
        
        # A simpler MILP: minimize sum of color indicators
        # x[v][c] binary for v in 0..n-1, c in 0..k-1
        # sum_c x[v][c] = 1 for all v
        # x[u][c] + x[v][c] <= 1 for all edges (u,v) and all c
        
        # For bicolored forest constraint, we need: for any pair (c1, c2),
        # the subgraph induced by colors c1 and c2 has no cycles.
        # This means for any cycle in the graph, not all vertices can have colors in {c1, c2}
        # AND at least one edge in the cycle must connect vertices of different colors
        # Actually, the constraint is: for any cycle, and any pair (c1,c2),
        # the cycle cannot be entirely contained in vertices colored c1 or c2.
        
        # So for each cycle C and each pair (c1, c2):
        # sum_{v in C} (x[v][c1] + x[v][c2]) <= |C| - 1
        # (i.e., not all vertices in the cycle have color c1 or c2)
        
        # Find all simple cycles up to length 6
        from collections import defaultdict
        
        def find_cycles(adj, max_len=6):
            cycles = set()
            n = len(adj)
            for start in range(n):
                stack = [(start, [start], {start})]
                while stack:
                    node, path, visited = stack.pop()
                    if len(path) > max_len:
                        continue
                    for nb in adj[node]:
                        if nb == start and len(path) >= 3:
                            # Found a cycle
                            cycle = tuple(sorted(path))
                            # Normalize: rotate so min vertex is first
                            min_idx = path.index(min(path))
                            norm = tuple(path[min_idx:] + path[:min_idx])
                            cycles.add(norm)
                        elif nb not in visited and nb > start:
                            # Only explore vertices > start to avoid duplicates
                            new_path = path + [nb]
                            new_visited = visited | {nb}
                            stack.append((nb, new_path, new_visited))
            return list(cycles)
        
        cycles = find_cycles(adj, max_len=6)
        
        # MILP formulation
        # x[v][c] for v in 0..n-1, c in 0..k-1
        # Total variables: n * k
        m = n * k
        c_obj = np.ones(m)  # Minimize number of colored vertices (but each vertex gets exactly one color)
        # Actually, we want to minimize k, but k is fixed here.
        # So we just need feasibility. Set objective to 0.
        c_obj = np.zeros(m)
        
        # Constraints:
        # 1. Each vertex gets exactly one color: sum_c x[v][c] = 1
        A_rows = []
        b_lower = []
        b_upper = []
        
        for v in range(n):
            row = np.zeros(m)
            for c in range(k):
                row[v * k + c] = 1
            A_rows.append(row)
            b_lower.append(1)
            b_upper.append(1)
        
        # 2. Adjacent vertices have different colors: x[u][c] + x[v][c] <= 1
        for u, v in edges:
            for c in range(k):
                row = np.zeros(m)
                row[u * k + c] = 1
                row[v * k + c] = 1
                A_rows.append(row)
                b_lower.append(0)
                b_upper.append(1)
        
        # 3. Bicolored forest constraint: for each cycle and each pair (c1, c2),
        #    sum_{v in cycle} (x[v][c1] + x[v][c2]) <= len(cycle) - 1
        for cycle in cycles:
            for c1 in range(k):
                for c2 in range(c1 + 1, k):
                    row = np.zeros(m)
                    for v in cycle:
                        row[v * k + c1] = 1
                        row[v * k + c2] = 1
                    A_rows.append(row)
                    b_lower.append(0)
                    b_upper.append(len(cycle) - 1)
        
        A = np.array(A_rows) if A_rows else np.zeros((1, m))
        b_lower = np.array(b_lower) if b_lower else np.zeros(1)
        b_upper = np.array(b_upper) if b_upper else np.zeros(1)
        
        integrality = np.ones(m)  # All variables are binary
        bounds = Bounds(lb=0, ub=1)
        
        try:
            res = milp(c=c_obj, constraints=LinearConstraint(A, b_lower, b_upper),
                       integrality=integrality, bounds=bounds)
            if res.status == 0:
                x = res.x.reshape(n, k)
                colors = [int(np.argmax(x[v])) for v in range(n)]
                if _validate(colors, edges):
                    return colors
        except Exception:
            pass
    
    # Fallback: use more colors
    for k in range(5, 12):
        colors = [-1] * n
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        def backtrack(idx):
            if idx == n:
                return True
            v = idx
            for c in range(k):
                valid = True
                for nb in adj[v]:
                    if colors[nb] == c:
                        valid = False
                        break
                if not valid:
                    continue
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                colors[v] = -1
            return False
        
        if backtrack(0):
            if _validate(colors, edges):
                return [int(c) for c in colors]
    
    # Last resort: use n colors (always valid)
    return list(range(n))
# EVOLVE-BLOCK-END
