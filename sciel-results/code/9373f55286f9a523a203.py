"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Vertex ordering by degree (descending)
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    
    def solve_with_k(k, deadline):
        """Backtracking with forward checking for bicolored forest constraint."""
        color = [-1] * n  # -1 means uncolored
        # DSU for each color pair: dsu[(c1,c2)] maps vertex -> parent
        # We'll use a dict of dicts for DSU
        # Actually, let's use a simpler approach: for each pair, maintain a set of edges and check connectivity
        # Better: use union-find per color pair
        
        # For efficiency, maintain for each color pair (c1, c2) a DSU
        # DSU stored as: dsu[pair_key] = {vertex: parent}
        # And also track which vertices are in each pair's subgraph
        
        from collections import defaultdict
        
        # For each color pair, we'll track:
        # - which vertices are in the subgraph
        # - union-find structure
        
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
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False  # cycle detected
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True
            return parent, rank, find, union
        
        # We'll store DSU state for each color pair
        # Key: (min(c1,c2), max(c1,c2))
        # Value: (parent_dict, rank_dict, find_func, union_func)
        # But functions can't be easily copied, so let's use a different approach
        
        # Alternative: maintain for each color pair a list of edges and check acyclicity
        # But that's O(E) per check which is slow.
        
        # Better approach: maintain DSU arrays
        # For each color pair, we need to track the DSU state.
        # Since we're doing backtracking, we need to be able to undo operations.
        
        # Let's use a different strategy: maintain for each color pair, a set of (u,v) edges
        # and check if the graph is a forest by counting edges vs vertices in each component.
        # Actually, the simplest correct check: for a graph to be a forest, 
        # for each connected component, edges = vertices - 1.
        # Equivalently, the total number of edges in the subgraph <= total vertices - number of components.
        
        # Most efficient: use DSU with path compression but track the number of successful unions.
        # A forest with v vertices has exactly v - c edges where c is number of components.
        # So we can track: for each color pair, (num_vertices, num_edges, num_components)
        # But we need to know if adding a specific edge creates a cycle.
        
        # Let me use the DSU approach with undo capability.
        # For backtracking, I'll save the DSU state before each assignment and restore after.
        
        # Actually, for 50 vertices and small k, let's just rebuild the check each time.
        # For each color pair, collect all edges between vertices of those two colors,
        # then check if it's a forest using BFS/DFS counting edges and vertices.
        
        def is_bicolored_forest(c1, c2, color_arr):
            """Check if subgraph induced by colors c1 and c2 is a forest."""
            # Collect vertices with color c1 or c2
            vertices = set()
            for i in range(n):
                if color_arr[i] == c1 or color_arr[i] == c2:
                    vertices.add(i)
            if len(vertices) <= 1:
                return True
            
            # Collect edges between these vertices
            edge_set = set()
            for u in vertices:
                for v in adj[u]:
                    if v in vertices and (u < v):
                        if color_arr[u] in (c1, c2) and color_arr[v] in (c1, c2):
                            edge_set.add((u, v))
            
            # A forest: for each connected component, edges = vertices - 1
            # Total: edges <= vertices - components
            # Check via BFS: count vertices and edges in each component
            visited = set()
            total_edges = 0
            total_vertices = 0
            num_components = 0
            
            for start in vertices:
                if start in visited:
                    continue
                # BFS
                queue = [start]
                visited.add(start)
                comp_vertices = 0
                comp_edges = 0
                while queue:
                    u = queue.pop()
                    comp_vertices += 1
                    for v in adj[u]:
                        if v in vertices:
                            comp_edges += 1
                            if v not in visited:
                                visited.add(v)
                                queue.append(v)
                comp_edges //= 2  # each edge counted twice
                total_edges += comp_edges
                total_vertices += comp_vertices
                num_components += 1
                if comp_edges > comp_vertices - 1:
                    return False
            
            return total_edges == total_vertices - num_components
        
        # This is O(V+E) per check, which for 50 vertices is fine.
        # But doing it for every pair at every step might be slow.
        # Let's optimize: only check pairs that are affected by the new assignment.
        
        def check_assignment(v, c, color_arr):
            """Check if assigning color c to vertex v is valid."""
            # Check adjacency constraint
            for u in adj[v]:
                if color_arr[u] == c:
                    return False
            
            # Check bicolored forest constraint for all existing colors
            existing_colors = set()
            for i in range(n):
                if color_arr[i] != -1:
                    existing_colors.add(color_arr[i])
            
            # Temporarily assign
            color_arr[v] = c
            
            # For each existing color c' != c, check pair (c, c')
            for c_prime in existing_colors:
                if c_prime == c:
                    continue
                if not is_bicolored_forest(c, c_prime, color_arr):
                    color_arr[v] = -1
                    return False
            
            # Also need to check: if c is new, we don't need to check (c, c) since
            # that's just the subgraph of color c alone, which is always a forest (no edges within same color... wait, 
            # actually there could be edges between two vertices both colored c? No! Adjacent vertices must differ.
            # So the subgraph induced by a single color has no edges, hence is a forest.
            
            color_arr[v] = -1
            return True
        
        # Backtracking
        def backtrack(idx):
            if time.time() > deadline:
                return None
            
            if idx == n:
                return color[:]
            
            v = order[idx]
            
            # Try colors in order, with some heuristic
            for c in range(k):
                if check_assignment(v, c, color):
                    color[v] = c
                    result = backtrack(idx + 1)
                    if result is not None:
                        return result
                    color[v] = -1
            
            return None
        
        return backtrack(0)
    
    # Try increasing k
    deadline = time.time() + 80  # Leave 10s margin
    best_result = None
    
    for k in range(3, 10):
        if time.time() > deadline:
            break
        result = solve_with_k(k, deadline)
        if result is not None:
            best_result = result
            break
    
    if best_result is None:
        # Fallback: use more colors
        # Simple greedy coloring
        best_result = [0] * n
        for v in order:
            used = set()
            for u in adj[v]:
                if best_result[u] != -1:
                    used.add(best_result[u])
            for c in range(50):
                if c not in used:
                    best_result[v] = c
                    break
    
    # Ensure all values are Python ints
    return [int(x) for x in best_result]
# EVOLVE-BLOCK-END
