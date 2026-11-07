"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

n = P['n']
edges = P['edges']

# Build adjacency list
adj = [[] for _ in range(n)]
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

# Precompute edge index per vertex for faster lookup
vertex_edges = [[] for _ in range(n)]
for i, (u, v) in enumerate(edges):
    vertex_edges[u].append(i)
    vertex_edges[v].append(i)

def count_distinct_colors(coloring):
    return len(set(coloring))

def validate(coloring):
    """Check all constraints: proper coloring and no bicolored cycles."""
    k = count_distinct_colors(coloring)
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    # Check no bicolored cycles using union-find
    for c1 in range(k):
        for c2 in range(c1 + 1, k):
            parent = list(range(n))
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
                cu, cv = coloring[u], coloring[v]
                if (cu == c1 or cu == c2) and (cv == c1 or cv == c2):
                    if not union(u, v):
                        return False
    return True

def backtrack_solve(k, deadline, color_order=None):
    """
    Backtracking solver with incremental constraint checking.
    Tries to find a valid k-coloring.
    """
    coloring = [-1] * n
    # Track which color pairs have been "seen" in the induced subgraph
    # For each pair (c1, c2), track the union-find state
    # But for efficiency, we'll check only when needed
    
    # Use a smarter ordering: vertices in order, but try colors in a fixed order
    if color_order is None:
        color_order = list(range(k))
    
    # Incremental check: when we assign color c to vertex v, we need to check:
    # 1. No neighbor has color c (proper coloring)
    # 2. For each neighbor u with color cu != c, the pair (c, cu) must not form a cycle
    #    This means: in the subgraph induced by colors {c, cu}, adding edge (v,u) 
    #    must not create a cycle.
    
    # For efficiency, maintain union-find per color pair
    # But that's complex. Instead, do a quick check:
    # When assigning color c to v, for each neighbor u with color cu:
    #   - if cu == c: fail (proper coloring)
    #   - else: check if edge (v,u) creates a cycle in the {c, cu} induced subgraph
    #     A cycle is created if v and u are already connected in the {c,cu} subgraph
    #     via other vertices.
    
    # For the check, we need to know if v and u are connected in the subgraph 
    # induced by colors {c, cu} using only already-assigned vertices.
    # We can do a BFS/DFS from v to u using only vertices colored c or cu.
    
    def would_create_cycle(v, c, u, cu):
        """Check if adding edge (v,u) creates a cycle in {c,cu} induced subgraph."""
        # BFS from v to u using only vertices with color c or cu (excluding the direct edge)
        # Actually, we need to check if there's a path from v to u in the {c,cu} subgraph
        # that doesn't use the edge (v,u) itself. Since we're adding this edge,
        # a cycle exists iff v and u are already connected via other {c,cu} vertices.
        
        target = cu
        # BFS from v
        visited = set()
        queue = [v]
        visited.add(v)
        while queue:
            curr = queue.pop(0)
            if curr == u:
                return True  # Path exists, so adding edge creates cycle
            for nb in adj[curr]:
                if nb in visited:
                    continue
                if coloring[nb] == c or coloring[nb] == cu:
                    visited.add(nb)
                    queue.append(nb)
        return False
    
    def solve(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        
        v = idx
        # Try each color
        for c in color_order:
            # Check proper coloring
            valid = True
            for nb in adj[v]:
                if coloring[nb] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check bicolored cycle constraint
            coloring[v] = c
            cycle_found = False
            for nb in adj[v]:
                if coloring[nb] != -1 and coloring[nb] != c:
                    if would_create_cycle(v, c, nb, coloring[nb]):
                        cycle_found = True
                        break
            
            if not cycle_found:
                if solve(idx + 1):
                    return True
            
            coloring[v] = -1
        
        return False
    
    if solve(0):
        return coloring
    return None

def backtrack_solve_with_order(k, deadline, vertex_order, color_order):
    """Backtracking with custom vertex and color ordering."""
    coloring = [-1] * n
    # Map from position in vertex_order to actual vertex
    # We'll work in the order given
    
    def would_create_cycle(v, c, u, cu, coloring):
        """Check if adding edge (v,u) creates a cycle in {c,cu} induced subgraph."""
        visited = set()
        queue = [v]
        visited.add(v)
        while queue:
            curr = queue.pop(0)
            if curr == u:
                return True
            for nb in adj[curr]:
                if nb in visited:
                    continue
                if coloring[nb] == c or coloring[nb] == cu:
                    visited.add(nb)
                    queue.append(nb)
        return False
    
    def solve(pos):
        if time.time() > deadline:
            return False
        if pos == n:
            return True
        
        v = vertex_order[pos]
        for c in color_order:
            # Check proper coloring
            valid = True
            for nb in adj[v]:
                if coloring[nb] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check bicolored cycle constraint
            coloring[v] = c
            cycle_found = False
            for nb in adj[v]:
                if coloring[nb] != -1 and coloring[nb] != c:
                    if would_create_cycle(v, c, nb, coloring[nb], coloring):
                        cycle_found = True
                        break
            
            if not cycle_found:
                if solve(pos + 1):
                    return True
            
            coloring[v] = -1
        
        return False
    
    if solve(0):
        return coloring
    return None

def construct():
    """
    Find a valid coloring of the degree-4 graph with minimum distinct colors
    such that no two colors form a bicolored cycle.
    """
    deadline = 80.0  # safety margin before 90s limit
    start = time.time()
    
    best_coloring = None
    best_k = n + 1
    
    # Try k from 3 upward
    for k in range(3, 12):
        if time.time() - start > deadline:
            break
        
        # Multiple strategies for vertex ordering
        strategies = []
        
        # Strategy 1: natural order
        strategies.append(list(range(n)))
        
        # Strategy 2: reverse natural order
        strategies.append(list(range(n - 1, -1, -1)))
        
        # Strategy 3: random order
        random.seed(42)
        strategies.append(random.sample(range(n), n))
        
        # Strategy 4: another random order
        random.seed(123)
        strategies.append(random.sample(range(n), n))
        
        # Strategy 5: BFS order from vertex 0
        visited = set()
        order = [0]
        visited.add(0)
        i = 0
        while i < len(order) and len(order) < n:
            v = order[i]
            i += 1
            for nb in adj[v]:
                if nb not in visited:
                    visited.add(nb)
                    order.append(nb)
        for v in range(n):
            if v not in visited:
                order.append(v)
        strategies.append(order)
        
        for vorder in strategies:
            if time.time() - start > deadline:
                break
            
            # Try different color orderings
            color_orders = [
                list(range(k)),
                list(range(k - 1, -1, -1)),
            ]
            random.seed(99)
            co = list(range(k))
            random.shuffle(co)
            color_orders.append(co)
            
            for corder in color_orders:
                if time.time() - start > deadline:
                    break
                
                coloring = backtrack_solve_with_order(k, deadline, vorder, corder)
                if coloring is not None:
                    k_used = count_distinct_colors(coloring)
                    if k_used < best_k:
                        best_k = k_used
                        best_coloring = list(coloring)
                    
                    # Found a solution with k colors, try to reduce further
                    # by trying k-1
                    break
            
            if best_k <= k:
                break
        
        if best_k <= k:
            break
    
    if best_coloring is not None:
        # Ensure all values are Python ints
        return [int(c) for c in best_coloring]
    
    # Fallback: all distinct colors
    return list(range(n))

# EVOLVE-BLOCK-END
