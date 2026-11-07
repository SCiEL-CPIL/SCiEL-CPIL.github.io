<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute edge list for quick lookup
    edge_set = set()
    for u, v in edges:
        if u < v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))
    
    def is_edge(u, v):
        if u == v:
            return False
        if u < v:
            return (u, v) in edge_set
        else:
            return (v, u) in edge_set
    
    # Union-Find for each color pair
    # We'll track components per color pair dynamically
    
    def solve():
        # Try with increasing number of colors
        for num_colors in range(3, 11):
            result = _solve_with_k_colors(num_colors)
            if result is not None:
                return result
        return None
    
    def _solve_with_k_colors(k):
        # Backtracking with forward checking
        # Order vertices by degree (descending)
        deg = [len(adj[i]) for i in range(n)]
        order = list(range(n))
        order.sort(key=lambda x: -deg[x])
        
        # Assignment array: -1 means unassigned
        assign = [-1] * n
        
        # For each color pair, we need to track connected components
        # Use union-find per color pair
        # Since k is small (3-10), we can have up to k*(k-1)/2 pairs
        # For each pair (c1, c2), maintain a DSU
        
        # We'll maintain:
        # dsu[pair_index] = dict mapping vertex -> parent (or use list)
        # Actually, let's use a simpler approach: for each pair, maintain a set of edges
        # and check for cycles using DFS when adding an edge
        
        # Better: maintain for each color pair a list of edges, and a DSU
        # When we assign a vertex, we add edges to all neighbors with the other color
        
        # Let's use a different approach:
        # For each color pair (c1, c2), maintain a DSU
        # When assigning vertex v to color c, for each neighbor u already assigned to color c' != c:
        #   If (c, c') pair exists, check if u and v are in the same component
        #   If yes, adding this edge would create a cycle -> reject
        #   If no, union them
        
        # DSU implementation
        def make_dsu():
            parent = {}
            rank = {}
            def find(x):
                if x not in parent:
                    parent[x] = x
                    rank[x] = 0
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                rx = find(x)
                ry = find(y)
                if rx == ry:
                    return False
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True
            def same(x, y):
                return find(x) == find(y)
            def reset():
                parent.clear()
                rank.clear()
            return {'find': find, 'union': union, 'same': same, 'reset': reset}
        
        # We'll use a single DSU per color pair, and track which vertices are in each
        # For backtracking, we need to be able to undo operations
        # This is complex. Instead, let's use a simpler validation:
        # After assigning a vertex, check all color pairs involving its color for cycles
        
        # Actually, let's use a cleaner backtracking with explicit state tracking
        
        # State: for each color pair, a list of edges (u, v) where both endpoints have those colors
        # To check if adding an edge creates a cycle, we can do a BFS/DFS on the subgraph
        
        # For efficiency with small k, let's precompute which vertices have which color
        # and maintain the edge lists per color pair
        
        # Let's try a different approach: greedy with local search, or just careful backtracking
        
        # Given the small size (50 vertices), let's do backtracking with the following:
        # - Assign vertices in order of decreasing degree
        # - For each vertex, try colors in order
        # - After assigning, validate all affected color pairs
        
        def validate_assignment(assign):
            """Check if the current partial assignment has any bicolored cycle."""
            # Group vertices by color
            color_to_vertices = defaultdict(list)
            for i in range(n):
                if assign[i] != -1:
                    color_to_vertices[assign[i]].append(i)
            
            # For each pair of colors, check if the induced subgraph is a forest
            colors = list(color_to_vertices.keys())
            for i in range(len(colors)):
                for j in range(i + 1, len(colors)):
                    c1 = colors[i]
                    c2 = colors[j]
                    # Get all vertices with color c1 or c2
                    verts = set(color_to_vertices[c1]) | set(color_to_vertices[c2])
                    # Build subgraph induced by these vertices
                    # Count edges and vertices
                    # A forest on m vertices has at most m - 1 edges
                    # More precisely, for each connected component, edges <= vertices - 1
                    
                    # Build adjacency for this subgraph
                    sub_adj = defaultdict(list)
                    edge_count = 0
                    for u in verts:
                        for v in adj[u]:
                            if v in verts and u < v:
                                sub_adj[u].append(v)
                                sub_adj[v].append(u)
                                edge_count += 1
                    
                    # Check if it's a forest using DFS
                    visited = set()
                    component_edges = {}
                    component_vertices = {}
                    for start in verts:
                        if start not in visited:
                            # BFS
                            queue = [start]
                            visited.add(start)
                            comp_verts = []
                            comp_edges = 0
                            while queue:
                                node = queue.pop()
                                comp_verts.append(node)
                                for neighbor in sub_adj[node]:
                                    if neighbor not in visited:
                                        visited.add(neighbor)
                                        queue.append(neighbor)
                                    comp_edges += 1
                            comp_edges //= 2  # Each edge counted twice
                            if len(comp_verts) > 0 and comp_edges > len(comp_verts) - 1:
                                return False
            return True
        
        # This validation is O(n^2) per step, which might be slow for backtracking
        # Let's optimize: only check color pairs that are affected by the new assignment
        
        def check_pair_forest(c1, c2, assign):
            """Check if the subgraph induced by colors c1 and c2 is a forest."""
            verts = set()
            for i in range(n):
                if assign[i] == c1 or assign[i] == c2:
                    verts.add(i)
            
            if len(verts) <= 1:
                return True
            
            # Build subgraph
            sub_adj = defaultdict(list)
            for u in verts:
                for v in adj[u]:
                    if v in verts and u < v:
                        sub_adj[u].append(v)
                        sub_adj[v].append(u)
            
            # Check for cycles using DFS
            visited = set()
            for start in verts:
                if start not in visited:
                    stack = [(start, -1, 0)]  # node, parent, state
                    visited.add(start)
                    comp_size = 0
                    comp_edge_count = 0
                    while stack:
                        node, parent, state = stack.pop()
                        if state == 0:
                            comp_size += 1
                            # Count edges
                            for neighbor in sub_adj[node]:
                                if neighbor != parent:
                                    comp_edge_count += 1
                            stack.append((node, parent, 1))
                            for neighbor in sub_adj[node]:
                                if neighbor not in visited:
                                    visited.add(neighbor)
                                    stack.append((neighbor, node, 0))
                                elif neighbor != parent:
                                    # Back edge, indicates cycle
                                    return False
                    comp_edge_count //= 2
                    if comp_edge_count > comp_size - 1:
                        return False
            return True
        
        # Now do backtracking
        def backtrack(idx):
            if idx == len(order):
                return True
            
            v = order[idx]
            
            for c in range(k):
                # Check if color c is valid for vertex v
                valid = True
                for u in adj[v]:
                    if assign[u] != -1 and assign[u] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Assign
                assign[v] = c
                
                # Check if this creates any bicolored cycle
                # Only need to check pairs involving color c
                ok = True
                for c2 in range(k):
                    if c2 == c:
                        continue
                    if not check_pair_forest(c, c2, assign):
                        ok = False
                        break
                
                if ok:
                    if backtrack(idx + 1):
                        return True
                
                # Undo
                assign[v] = -1
            
            return False
        
        if backtrack(0):
            return assign
        return None
    
    result = solve()
    if result is None:
        # Fallback: return unique colors (always valid but not optimal)
        result = list(range(n))
    
    # Ensure all are Python ints
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute edge list for quick lookup
    edge_set = set()
    for u, v in edges:
        if u < v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))
    
    def is_edge(u, v):
        if u == v:
            return False
        if u < v:
            return (u, v) in edge_set
        else:
            return (v, u) in edge_set
    
    # Union-Find for each color pair
    # We'll track components per color pair dynamically
    
    def solve():
        # Try with increasing number of colors
        for num_colors in range(3, 11):
            result = _solve_with_k_colors(num_colors)
            if result is not None:
                return result
        return None
    
    def _solve_with_k_colors(k):
        # Backtracking with forward checking
        # Order vertices by degree (descending)
        deg = [len(adj[i]) for i in range(n)]
        order = list(range(n))
        order.sort(key=lambda x: -deg[x])
        
        # Assignment array: -1 means unassigned
        assign = [-1] * n
        
        # For each color pair, we need to track connected components
        # Use union-find per color pair
        # Since k is small (3-10), we can have up to k*(k-1)/2 pairs
        # For each pair (c1, c2), maintain a DSU
        
        # We'll maintain:
        # dsu[pair_index] = dict mapping vertex -> parent (or use list)
        # Actually, let's use a simpler approach: for each pair, maintain a set of edges
        # and check for cycles using DFS when adding an edge
        
        # Better: maintain for each color pair a list of edges, and a DSU
        # When we assign a vertex, we add edges to all neighbors with the other color
        
        # Let's use a different approach:
        # For each color pair (c1, c2), maintain a DSU
        # When assigning vertex v to color c, for each neighbor u already assigned to color c' != c:
        #   If (c, c') pair exists, check if u and v are in the same component
        #   If yes, adding this edge would create a cycle -> reject
        #   If no, union them
        
        # DSU implementation
        def make_dsu():
            parent = {}
            rank = {}
            def find(x):
                if x not in parent:
                    parent[x] = x
                    rank[x] = 0
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                rx = find(x)
                ry = find(y)
                if rx == ry:
                    return False
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True
            def same(x, y):
                return find(x) == find(y)
            def reset():
                parent.clear()
                rank.clear()
            return {'find': find, 'union': union, 'same': same, 'reset': reset}
        
        # We'll use a single DSU per color pair, and track which vertices are in each
        # For backtracking, we need to be able to undo operations
        # This is complex. Instead, let's use a simpler validation:
        # After assigning a vertex, check all color pairs involving its color for cycles
        
        # Actually, let's use a cleaner backtracking with explicit state tracking
        
        # State: for each color pair, a list of edges (u, v) where both endpoints have those colors
        # To check if adding an edge creates a cycle, we can do a BFS/DFS on the subgraph
        
        # For efficiency with small k, let's precompute which vertices have which color
        # and maintain the edge lists per color pair
        
        # Let's try a different approach: greedy with local search, or just careful backtracking
        
        # Given the small size (50 vertices), let's do backtracking with the following:
        # - Assign vertices in order of decreasing degree
        # - For each vertex, try colors in order
        # - After assigning, validate all affected color pairs
        
        def validate_assignment(assign):
            """Check if the current partial assignment has any bicolored cycle."""
            # Group vertices by color
            color_to_vertices = defaultdict(list)
            for i in range(n):
                if assign[i] != -1:
                    color_to_vertices[assign[i]].append(i)
            
            # For each pair of colors, check if the induced subgraph is a forest
            colors = list(color_to_vertices.keys())
            for i in range(len(colors)):
                for j in range(i + 1, len(colors)):
                    c1 = colors[i]
                    c2 = colors[j]
                    # Get all vertices with color c1 or c2
                    verts = set(color_to_vertices[c1]) | set(color_to_vertices[c2])
                    # Build subgraph induced by these vertices
                    # Count edges and vertices
                    # A forest on m vertices has at most m - 1 edges
                    # More precisely, for each connected component, edges <= vertices - 1
                    
                    # Build adjacency for this subgraph
                    sub_adj = defaultdict(list)
                    edge_count = 0
                    for u in verts:
                        for v in adj[u]:
                            if v in verts and u < v:
                                sub_adj[u].append(v)
                                sub_adj[v].append(u)
                                edge_count += 1
                    
                    # Check if it's a forest using DFS
                    visited = set()
                    component_edges = {}
                    component_vertices = {}
                    for start in verts:
                        if start not in visited:
                            # BFS
                            queue = [start]
                            visited.add(start)
                            comp_verts = []
                            comp_edges = 0
                            while queue:
                                node = queue.pop()
                                comp_verts.append(node)
                                for neighbor in sub_adj[node]:
                                    if neighbor not in visited:
                                        visited.add(neighbor)
                                        queue.append(neighbor)
                                    comp_edges += 1
                            comp_edges //= 2  # Each edge counted twice
                            if len(comp_verts) > 0 and comp_edges > len(comp_verts) - 1:
                                return False
            return True
        
        # This validation is O(n^2) per step, which might be slow for backtracking
        # Let's optimize: only check color pairs that are affected by the new assignment
        
        def check_pair_forest(c1, c2, assign):
            """Check if the subgraph induced by colors c1 and c2 is a forest."""
            verts = set()
            for i in range(n):
                if assign[i] == c1 or assign[i] == c2:
                    verts.add(i)
            
            if len(verts) <= 1:
                return True
            
            # Build subgraph
            sub_adj = defaultdict(list)
            for u in verts:
                for v in adj[u]:
                    if v in verts and u < v:
                        sub_adj[u].append(v)
                        sub_adj[v].append(u)
            
            # Check for cycles using DFS
            visited = set()
            for start in verts:
                if start not in visited:
                    stack = [(start, -1, 0)]  # node, parent, state
                    visited.add(start)
                    comp_size = 0
                    comp_edge_count = 0
                    while stack:
                        node, parent, state = stack.pop()
                        if state == 0:
                            comp_size += 1
                            # Count edges
                            for neighbor in sub_adj[node]:
                                if neighbor != parent:
                                    comp_edge_count += 1
                            stack.append((node, parent, 1))
                            for neighbor in sub_adj[node]:
                                if neighbor not in visited:
                                    visited.add(neighbor)
                                    stack.append((neighbor, node, 0))
                                elif neighbor != parent:
                                    # Back edge, indicates cycle
                                    return False
                    comp_edge_count //= 2
                    if comp_edge_count > comp_size - 1:
                        return False
            return True
        
        # Now do backtracking
        def backtrack(idx):
            if idx == len(order):
                return True
            
            v = order[idx]
            
            for c in range(k):
                # Check if color c is valid for vertex v
                valid = True
                for u in adj[v]:
                    if assign[u] != -1 and assign[u] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Assign
                assign[v] = c
                
                # Check if this creates any bicolored cycle
                # Only need to check pairs involving color c
                ok = True
                for c2 in range(k):
                    if c2 == c:
                        continue
                    if not check_pair_forest(c, c2, assign):
                        ok = False
                        break
                
                if ok:
                    if backtrack(idx + 1):
                        return True
                
                # Undo
                assign[v] = -1
            
            return False
        
        if backtrack(0):
            return assign
        return None
    
    result = solve()
    if result is None:
        # Fallback: return unique colors (always valid but not optimal)
        result = list(range(n))
    
    # Ensure all are Python ints
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I need to actually implement the optimized version. The previous block was just copying the old code. I will replace the `construct` function with the optimized Union-Find based backtracking.

