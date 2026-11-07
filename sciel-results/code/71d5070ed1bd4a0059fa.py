"""Solver for few-color graph coloring with no bicolored cycles."""
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

def _check_valid(coloring, adj):
    """Check adjacent differ and no bicolored cycles. Returns True if valid."""
    n = len(coloring)
    # Check adjacent vertices differ
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False
    # Check no bicolored cycles using union-find per color pair
    color_pairs = set()
    for u, v in P['edges']:
        cu, cv = coloring[u], coloring[v]
        color_pairs.add((min(cu, cv), max(cu, cv)))
    
    for c1, c2 in color_pairs:
        parent = list(range(n))
        rank = [0] * n
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            return True
        for u, v in P['edges']:
            cu, cv = coloring[u], coloring[v]
            if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                if not union(u, v):
                    return False
    return True

def _incremental_check(coloring, adj, new_vertex, new_color, edge_list):
    """Fast check: only verify constraints involving new_vertex and new_color interactions."""
    n = len(coloring)
    # Check new vertex doesn't conflict with neighbors
    for nb in adj[new_vertex]:
        if coloring[nb] == new_color:
            return False
    
    # Check bicolored cycles: for each color c that new_color pairs with,
    # check if adding new_vertex's edges creates a cycle
    # Get colors of neighbors
    neighbor_colors = set()
    for nb in adj[new_vertex]:
        neighbor_colors.add(coloring[nb])
    
    for c in neighbor_colors:
        if c == new_color:
            continue
        c1, c2 = min(new_color, c), max(new_color, c)
        # Check if there's already a path between any two neighbors of new_vertex
        # that have colors c1 and c2 (or vice versa)
        # Actually, we need to check: among neighbors of new_vertex, 
        # are there two vertices u, v such that coloring[u] in {c1,c2} and coloring[v] in {c1,c2}
        # and there's a path from u to v using only edges between vertices colored c1 or c2?
        # Simpler: just do a full check for this color pair on the partial coloring
        # But that's expensive. Let's do a targeted BFS.
        
        # Collect all edges in the current partial coloring that involve colors c1 and c2
        # and are incident to new_vertex or connect two vertices both colored c1/c2
        # Actually, the simplest correct approach: do union-find just for this pair
        parent = {}
        def find(x):
            if x not in parent:
                parent[x] = x
                return x
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            parent[rb] = ra
            return True
        
        # Add all edges between vertices colored c1 or c2 in the current partial coloring
        for u, v in edge_list:
            cu, cv = coloring[u], coloring[v]
            if cu == -1 or cv == -1:
                continue
            if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                if not union(u, v):
                    return False
            elif cu == cv == c1 or cu == cv == c2:
                # Same color edge - not a bicolored edge, skip
                pass
        
        # Now add new_vertex's edges to neighbors with color c
        for nb in adj[new_vertex]:
            if coloring[nb] == c:
                if not union(new_vertex, nb):
                    return False
        # Also add edges from new_vertex to neighbors with color new_color? No, those aren't bicolored with c
        # Wait, new_vertex has color new_color, so edges to neighbors with color c ARE bicolored (c1,c2)
        # We already added those above.
        
        # But we also need to check: are there other neighbors of new_vertex with color c1 or c2?
        # new_vertex has color new_color = one of {c1, c2}
        # Edges from new_vertex to neighbors with the OTHER color are bicolored
        other_c = c2 if new_color == c1 else c1
        for nb in adj[new_vertex]:
            if coloring[nb] == other_c:
                if not union(new_vertex, nb):
                    return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    adj = _build_adj()
    n = P['n']
    edge_list = P['edges']
    
    # Try systematic patterns first - these are O(1) to check
    patterns = []
    
    # v mod k
    for k in [3, 4, 5, 6, 7, 8]:
        patterns.append([v % k for v in range(n)])
    
    # (v // 10 + v % 10) % k
    for k in [3, 4, 5, 6]:
        patterns.append([(v // 10 + v % 10) % k for v in range(n)])
    
    # (v // 5) % k
    for k in [3, 4, 5, 6]:
        patterns.append([(v // 5) % k for v in range(n)])
    
    # (v // 10) * 2 + (v % 2)
    patterns.append([(v // 10) * 2 + (v % 2) for v in range(n)])
    
    # v % 4 + 4 * (v // 10 % 2)
    patterns.append([(v % 4) + 4 * ((v // 10) % 2) for v in range(n)])
    
    # Try each pattern
    for coloring in patterns:
        if time.time() > deadline:
            break
        if _check_valid(coloring, adj):
            return [int(x) for x in coloring]
    
    # Now try backtracking with incremental checking
    # Use a better vertex ordering: most constrained first
    # All vertices have degree 4, so use a heuristic: 
    # order by the number of already-colored neighbors (dynamic)
    # For simplicity, use a static order based on position
    
    # Let's try a different approach: use local search / simulated annealing
    # Start with a random valid-ish coloring and improve
    
    import random
    random.seed(42)
    
    # Start with a 4-coloring pattern and try to fix violations
    # First, let's try a smarter backtracking with better pruning
    
    # Use DSATUR-like heuristic: always color the vertex with most colored neighbors first
    # among uncolored vertices
    
    def solve_with_k_colors(k, time_limit):
        """Backtracking with incremental bicolored cycle check."""
        coloring = [-1] * n
        adj_set = [set(a) for a in adj]
        
        # Precompute for fast incremental checks
        # For each color pair, maintain a union-find structure
        # This is complex; instead, do targeted checks
        
        # Order: start with vertex 0, then pick next vertex with most colored neighbors
        assigned = [False] * n
        assigned[0] = True
        coloring[0] = 0
        
        # Check if initial assignment is valid (trivially yes for single vertex)
        
        def get_unassigned_order():
            """Return unassigned vertices sorted by number of assigned neighbors (descending)."""
            unassigned = []
            for v in range(n):
                if not assigned[v]:
                    num_assigned_nb = sum(1 for nb in adj[v] if assigned[nb])
                    unassigned.append((num_assigned_nb, v))
            unassigned.sort(reverse=True)
            return [v for _, v in unassigned]
        
        def is_consistent():
            """Check full consistency of current partial coloring."""
            for u in range(n):
                if not assigned[u]:
                    continue
                for v in adj[u]:
                    if assigned[v] and coloring[u] == coloring[v]:
                        return False
            # Check bicolored cycles
            color_pairs = set()
            for u, v in edge_list:
                if assigned[u] and assigned[v]:
                    cu, cv = coloring[u], coloring[v]
                    if cu != cv:
                        color_pairs.add((min(cu, cv), max(cu, cv)))
            
            for c1, c2 in color_pairs:
                parent = list(range(n))
                rank = [0] * n
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                def union(a, b):
                    ra, rb = find(a), find(b)
                    if ra == rb:
                        return False
                    if rank[ra] < rank[rb]:
                        ra, rb = rb, ra
                    parent[rb] = ra
                    if rank[ra] == rank[rb]:
                        rank[ra] += 1
                    return True
                for u, v in edge_list:
                    if not (assigned[u] and assigned[v]):
                        continue
                    cu, cv = coloring[u], coloring[v]
                    if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                        if not union(u, v):
                            return False
            return True
        
        def dfs(remaining):
            if time.time() > time_limit:
                return None
            if remaining == 0:
                return coloring[:]
            
            # Pick next vertex: most constrained
            best_v = -1
            best_score = -1
            for v in range(n):
                if not assigned[v]:
                    num_assigned_nb = sum(1 for nb in adj[v] if assigned[nb])
                    if num_assigned_nb > best_score:
                        best_score = num_assigned_nb
                        best_v = v
            
            if best_v == -1:
                return None
            
            # Try each color
            used_colors = set()
            for nb in adj[best_v]:
                if assigned[nb]:
                    used_colors.add(coloring[nb])
            
            for c in range(k):
                if c in used_colors:
                    continue
                coloring[best_v] = c
                assigned[best_v] = True
                
                # Quick check: does this create a bicolored cycle?
                # Check for each neighbor color
                valid = True
                for nb in adj[best_v]:
                    if assigned[nb]:
                        c_nb = coloring[nb]
                        if c_nb == c:
                            continue
                        c1, c2 = min(c, c_nb), max(c, c_nb)
                        # Check if adding this edge creates a cycle in the c1-c2 subgraph
                        # Do a BFS from best_v to nb using only vertices colored c1 or c2
                        # (excluding the direct edge)
                        visited = {best_v}
                        stack = [best_v]
                        found_path = False
                        while stack:
                            u = stack.pop()
                            for w in adj[u]:
                                if w == nb:
                                    # Found a path from best_v to nb
                                    # But we need to check if this creates a cycle
                                    # The edge (best_v, nb) plus a path from best_v to nb
                                    # means there's a cycle if the path doesn't use the edge
                                    # Actually, if there's ANY path from best_v to nb using only
                                    # c1/c2 colored vertices (other than the direct edge),
                                    # then adding the direct edge creates a cycle
                                    found_path = True
                                    break
                                if w in visited:
                                    continue
                                if not assigned[w]:
                                    continue
                                cw = coloring[w]
                                if cw == c1 or cw == c2:
                                    visited.add(w)
                                    stack.append(w)
                            if found_path:
                                break
                        if found_path:
                            valid = False
                            break
                
                if valid:
                    result = dfs(remaining - 1)
                    if result is not None:
                        return result
                
                assigned[best_v] = False
                coloring[best_v] = -1
            
            return None
        
        return dfs(n - 1)
    
    # Try 3 colors first (most constrained)
    if time.time() < deadline - 10:
        result = solve_with_k_colors(3, time.time() + 5)
        if result is not None:
            if _check_valid(result, adj):
                return [int(x) for x in result]
    
    # Try 4 colors
    if time.time() < deadline - 10:
        result = solve_with_k_colors(4, time.time() + 10)
        if result is not None:
            if _check_valid(result, adj):
                return [int(x) for x in result]
    
    # Try 5 colors
    if time.time() < deadline - 5:
        result = solve_with_k_colors(5, time.time() + 5)
        if result is not None:
            if _check_valid(result, adj):
                return [int(x) for x in result]
    
    # Fallback: 50 distinct colors
    return list(range(n))
