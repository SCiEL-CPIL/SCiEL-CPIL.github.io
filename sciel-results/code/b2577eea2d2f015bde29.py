"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency():
    n = P['n']
    adj = defaultdict(set)
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _has_bicolored_cycle(coloring, color_a, color_b):
    """Check if the subgraph induced by color_a and color_b has a cycle using union-find."""
    n = P['n']
    parent = list(range(n))
    rank = [0] * n
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False  # cycle detected
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True
    
    for u, v in P['edges']:
        cu, cv = coloring[u], coloring[v]
        if cu == color_a and cv == color_b:
            if not union(u, v):
                return True
        elif cu == color_b and cv == color_a:
            if not union(u, v):
                return True
    return False

def _is_valid(coloring):
    """Check if coloring is valid: adjacent vertices differ, and no bicolored cycle."""
    n = P['n']
    # Check adjacent vertices differ
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False
    # Check no bicolored cycles
    colors_used = set(coloring)
    for i, ca in enumerate(colors_used):
        for cb in list(colors_used)[i+1:]:
            if _has_bicolored_cycle(coloring, ca, cb):
                return False
    return True

def _count_distinct(coloring):
    return len(set(coloring))

def _backtrack_search(adj, max_colors, deadline):
    """Backtracking search to find a valid coloring with at most max_colors colors."""
    n = P['n']
    coloring = [-1] * n
    
    # Order vertices by degree (descending) for better pruning
    vertices_by_degree = sorted(range(n), key=lambda v: -len(adj[v]))
    v_to_idx = {v: i for i, v in enumerate(vertices_by_degree)}
    
    # Precompute: for each vertex, which vertices come before it in ordering
    # We'll assign in the order vertices_by_degree
    
    def backtrack(idx):
        if time.time() > deadline:
            return None
        if idx == n:
            return coloring[:]
        
        v = vertices_by_degree[idx]
        
        # Try each color
        for c in range(max_colors):
            # Check if color c is allowed (adjacent vertices must differ)
            conflict = False
            for nb in adj[v]:
                nb_idx = v_to_idx[nb]
                if nb_idx < idx:  # already assigned
                    if coloring[nb] == c:
                        conflict = True
                        break
            if conflict:
                continue
            
            coloring[v] = c
            
            # Check bicolored cycle constraint incrementally
            valid = True
            for nb in adj[v]:
                nb_idx = v_to_idx[nb]
                if nb_idx < idx:  # already assigned
                    cnb = coloring[nb]
                    if cnb == c:
                        valid = False
                        break
                    # Check if adding this edge creates a bicolored cycle
                    # The bicolored subgraph for (c, cnb) now includes edge (v, nb)
                    # We need to check if there's already a path between v and nb
                    # in the bicolored subgraph for (c, cnb)
                    # This is expensive, so let's do a quick check
                    # Actually, let's just check at the end for simplicity,
                    # or do a more efficient incremental check
                    pass
            
            if valid:
                # Quick check: for each color pair (c, cnb) where cnb is used by a neighbor,
                # check if the bicolored subgraph has a cycle
                # This is still expensive, so let's use a different approach:
                # Only check when we've assigned all vertices, or use a smarter method
                result = backtrack(idx + 1)
                if result is not None:
                    return result
            
            coloring[v] = -1
        
        return None
    
    return backtrack(0)

def _backtrack_search_full(adj, max_colors, deadline):
    """Backtracking with full validation at each step (slower but correct)."""
    n = P['n']
    coloring = [-1] * n
    
    vertices_by_degree = sorted(range(n), key=lambda v: -len(adj[v]))
    v_to_idx = {v: i for i, v in enumerate(vertices_by_degree)}
    
    # For incremental bicolored cycle checking, maintain union-find per color pair
    # This is complex, so let's use a simpler approach: check validity after each assignment
    # but only check the color pairs involving the new vertex
    
    def check_bicolored_cycle_for_new_edge(v, c, nb, cnb):
        """Check if adding edge (v, nb) with colors (c, cnb) creates a bicolored cycle.
        We need to check if there's a path between v and nb in the subgraph induced by colors c and cnb,
        excluding the edge (v, nb) itself."""
        # BFS/DFS from v to nb in the bicolored subgraph
        if v == nb:
            return True
        visited = {v}
        stack = [v]
        while stack:
            cur = stack.pop()
            for nb2 in adj[cur]:
                if nb2 == nb and cur == v:
                    # This would be the edge we're adding, skip for the "other path" check
                    # Actually we want to know if there's ANOTHER path
                    pass
                cnb2 = coloring[nb2]
                if cnb2 in (c, cnb) and nb2 not in visited:
                    if nb2 == nb:
                        # Found a path, but need to make sure it's not just the direct edge
                        # If cur == v and nb2 == nb, that's the direct edge
                        if not (cur == v and nb2 == nb):
                            return True
                        # If cur == nb and nb2 == v, that's also the direct edge
                        if not (cur == nb and nb2 == v):
                            return True
                    visited.add(nb2)
                    stack.append(nb2)
        return False
    
    def backtrack(idx):
        if time.time() > deadline:
            return None
        if idx == n:
            return coloring[:]
        
        v = vertices_by_degree[idx]
        
        for c in range(max_colors):
            # Check adjacent vertices differ
            conflict = False
            for nb in adj[v]:
                nb_idx = v_to_idx[nb]
                if nb_idx < idx:
                    if coloring[nb] == c:
                        conflict = True
                        break
            if conflict:
                continue
            
            coloring[v] = c
            
            # Check bicolored cycle constraints for edges from v to already-assigned neighbors
            valid = True
            for nb in adj[v]:
                nb_idx = v_to_idx[nb]
                if nb_idx < idx:
                    cnb = coloring[nb]
                    # Check if adding edge (v, nb) creates a bicolored cycle
                    if check_bicolored_cycle_for_new_edge(v, c, nb, cnb):
                        valid = False
                        break
            
            if valid:
                result = backtrack(idx + 1)
                if result is not None:
                    return result
            
            coloring[v] = -1
        
        return None
    
    return backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    adj = _build_adjacency()
    n = P['n']
    
    # Try increasing number of colors starting from a lower bound
    # The graph has max degree 4, so chromatic number is at most 5
    # But we also need the bicolored cycle constraint
    
    # Start with 4 colors (since max degree is 4, 4-coloring might work)
    # Actually, let's try from 3 first since it's a sparse graph
    
    best_coloring = None
    best_num_colors = n + 1
    
    for num_colors in range(3, n + 1):
        if time.time() > deadline:
            break
        if num_colors >= best_num_colors:
            break
            
        result = _backtrack_search_full(adj, num_colors, deadline)
        if result is not None:
            best_coloring = result
            best_num_colors = num_colors
            break
    
    if best_coloring is None:
        # Fallback: use all distinct colors (always valid)
        best_coloring = list(range(n))
    
    # Ensure all values are Python ints
    best_coloring = [int(c) for c in best_coloring]
    
    return best_coloring
# EVOLVE-BLOCK-END
