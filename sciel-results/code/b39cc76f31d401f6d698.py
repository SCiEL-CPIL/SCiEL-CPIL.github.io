"""Rewritten program: smart backtracking with forest constraint pruning."""
import time
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute all cycles of length 3-8 for constraint checking
    # A bicolored cycle exists if some cycle in the graph has all vertices colored with only 2 colors
    # We'll use the approach: for each pair of colors (a,b), check if subgraph of vertices with color a or b has a cycle
    
    # Vertex ordering: use a heuristic - order by degree descending, then by index
    # Actually, let's use a smarter ordering based on the graph structure
    # The graph looks like it has a "ladder" structure with diagonals
    
    # Let me identify the structure:
    # Main path: 0-1-2-3-4-5-6-7-8-...-49 (path edges)
    # Plus "diagonal" edges: 0-7, 1-8, 2-9, ..., 42-49
    # Plus "outer" edges: 0-43, 1-44, 2-45, ..., 5-48, 6-49, 36-43, 37-44, 38-45, 39-46, 40-47, 41-48
    
    # This is a 4-regular graph on 50 vertices
    
    # Strategy: Try k=3,4,5,... colors
    # For each k, use backtracking with strong pruning
    
    best_result = None
    
    def check_bicolored_forest(coloring, n):
        """Check that for every pair of colors, the induced subgraph is a forest."""
        # Group vertices by color
        color_groups = defaultdict(list)
        for i in range(n):
            color_groups[coloring[i]].append(i)
        
        colors_used = list(color_groups.keys())
        for i in range(len(colors_used)):
            for j in range(i+1, len(colors_used)):
                a, b = colors_used[i], colors_used[j]
                # Get vertices with color a or b
                verts = set(color_groups[a]) | set(color_groups[b])
                if len(verts) <= 1:
                    continue
                # Check if the induced subgraph on these vertices is a forest
                # Use union-find to detect cycles
                parent = {}
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                def union(x, y):
                    px, py = find(x), find(y)
                    if px == py:
                        return False  # cycle detected
                    parent[px] = py
                    return True
                
                for v in verts:
                    parent[v] = v
                
                for v in verts:
                    for u in adj[v]:
                        if u in verts and u > v:  # avoid double counting
                            if not union(v, u):
                                return False
        return True
    
    def backtrack(v, coloring, k, color_counts, adj_conflicts, deadline):
        """Backtracking search for valid coloring with k colors."""
        if time.time() > deadline:
            return None
        
        if v == n:
            # Full coloring found, verify
            return list(coloring)
        
        # Try each color for vertex v
        for c in range(k):
            # Check adjacency constraint
            valid = True
            for u in adj[v]:
                if coloring[u] == c:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check bicolored forest constraint incrementally
            # For each existing color d != c, check if adding v with color c
            # to the (c,d) subgraph creates a cycle
            # We need to track the forest structure for each color pair
            
            coloring[v] = c
            result = backtrack(v + 1, coloring, k, color_counts, adj_conflicts, deadline)
            if result is not None:
                return result
            coloring[v] = -1
        
        return None
    
    # Better approach: use union-find state tracking for each color pair
    # This is complex, so let's use a simpler but effective approach:
    # 1. Use a good vertex ordering
    # 2. At each step, check if the partial coloring can be extended
    # 3. Periodically verify the full constraint
    
    # Let's use a different strategy: 
    # - Order vertices in a specific sequence that makes the constraint checking efficient
    # - Use forward checking
    
    # Vertex ordering: try to put vertices with many constraints early
    # Compute a good ordering using greedy coloring heuristic
    order = list(range(n))
    # Sort by degree descending, then by index
    order.sort(key=lambda x: (-len(adj[x]), x))
    
    # Actually, for this specific graph structure, let me try a specific ordering
    # that respects the graph's natural structure
    
    # Let me try: first color vertices 0-6 (the "top" part), then work down
    # Actually, let's just use a systematic approach with good pruning
    
    # Key insight: if we use 3 colors, each color class must be an independent set
    # AND for any two colors, their union must form a forest
    # This is equivalent to: the graph is 3-colorable AND no two color classes together contain a cycle
    
    # Let's try k=3 first with a well-structured backtracking
    
    def solve_with_k(k, deadline):
        """Try to find a valid k-coloring."""
        # Use the natural vertex order 0,1,2,...,n-1
        # But with forward checking on the forest constraint
        
        coloring = [-1] * n
        
        # For efficient bicolored cycle checking, maintain for each color pair (a,b)
        # the connected components using union-find
        # But this is expensive to maintain incrementally...
        
        # Alternative: use a simpler check at each step
        # When assigning color c to vertex v, for each other color d that exists,
        # check if the (c,d)-induced subgraph would have a cycle
        
        # For efficiency, let's precompute: for each vertex v and color c,
        # which other vertices have colors that would conflict
        
        # Actually, let's use a recursive approach with memoization on the
        # "state" being which vertices are colored and their colors
        
        # Given the size (50 vertices), let's use a practical approach:
        # 1. Try a greedy assignment
        # 2. If it fails, use limited backtracking
        
        # Greedy with backtracking
        def dfs(pos, coloring):
            if time.time() > deadline:
                return None
            if pos == n:
                return list(coloring)
            
            v = pos  # vertex to color
            
            # Determine available colors
            available = set(range(k))
            for u in adj[v]:
                if coloring[u] != -1:
                    available.discard(coloring[u])
            
            if not available:
                return None
            
            # For each candidate color, check if it maintains the forest property
            for c in sorted(available):
                coloring[v] = c
                
                # Check bicolored forest property incrementally
                # For each color d that has at least one vertex, check (c,d) subgraph
                # Only need to check if v's addition creates a cycle
                valid = True
                for d in range(k):
                    if d == c:
                        continue
                    # Check if adding v (color c) to the (c,d) subgraph creates a cycle
                    # Find all neighbors of v that have color d
                    d_neighbors = [u for u in adj[v] if coloring[u] == d]
                    if not d_neighbors:
                        continue
                    # Also check c-colored neighbors of v (excluding v itself)
                    c_neighbors = [u for u in adj[v] if coloring[u] == c]
                    if not c_neighbors:
                        continue
                    # Now check if there's a path between any two of these neighbors
                    # in the (c,d)-induced subgraph (excluding v)
                    # This is expensive... let's use union-find on the current (c,d) subgraph
                    
                    # For efficiency, let's just do a BFS/DFS from each d_neighbor
                    # to see if we can reach another d_neighbor or c_neighbor
                    # through only c and d colored vertices (excluding v)
                    
                    visited = set()
                    # Start from all c_neighbors and d_neighbors, see if they're connected
                    # through the (c,d) subgraph
                    
                    # Actually, a cycle is created iff there exist two neighbors of v
                    # (one with color c, one with color d, or two with same color)
                    # that are connected in the (c,d)-subgraph excluding v
                    
                    # Let's check: is there a path between any pair of v's neighbors
                    # in the subgraph induced by vertices with color c or d, excluding v?
                    
                    # Collect all neighbors of v that have color c or d
                    relevant_neighbors = [u for u in adj[v] if coloring[u] in (c, d)]
                    if len(relevant_neighbors) < 2:
                        continue
                    
                    # Check if any two of these neighbors are connected in the (c,d) subgraph
                    # (excluding v)
                    # Use BFS from the first relevant neighbor
                    start = relevant_neighbors[0]
                    visited = {start}
                    queue = [start]
                    found_connection = False
                    while queue and not found_connection:
                        curr = queue.pop(0)
                        for nb in adj[curr]:
                            if nb == v:
                                continue
                            if nb in visited:
                                continue
                            if coloring[nb] == -1 or coloring[nb] not in (c, d):
                                continue
                            if nb in relevant_neighbors and nb != start:
                                found_connection = True
                                break
                            visited.add(nb)
                            queue.append(nb)
                    
                    if found_connection:
                        valid = False
                        break
                
                if valid:
                    result = dfs(pos + 1, coloring)
                    if result is not None:
                        return result
                
                coloring[v] = -1
            
            return None
        
        return dfs(0, coloring)
    
    # Try increasing k
    for k in range(3, 8):
        if time.time() > deadline:
            break
        result = solve_with_k(k, deadline)
        if result is not None:
            # Verify the solution
            if check_bicolored_forest(result, n):
                return result
            else:
                # Should not happen if our incremental check is correct
                continue
    
    # Fallback: return a valid coloring with more colors
    # Use greedy coloring with a good vertex ordering
    coloring = [0] * n
    for v in range(n):
        used = set()
        for u in adj[v]:
            if u < v:
                used.add(coloring[u])
        for c in range(50):
            if c not in used:
                coloring[v] = c
                break
    
    # This greedy might not satisfy the forest constraint, so let's verify
    # and if needed, use a fallback with more colors
    if check_bicolored_forest(coloring, n):
        return coloring
    
    # Ultimate fallback: all different colors
    return list(range(n))

# EVOLVE-BLOCK-END
