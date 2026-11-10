"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Order vertices for backtracking (e.g., descending degree)
    degrees = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    # colors[i] = -1 if unassigned, else 0, 1, 2
    colors = [-1] * n
    
    # Helper to check if adding vertex v with color c creates a bicolored cycle
    # with any other color c'
    def creates_cycle(v, c, c_prime):
        # We need to check if the subgraph induced by {v} U {u | colors[u] == c_prime} has a cycle.
        # This is equivalent to checking if v has two or more neighbors in the same connected component
        # of the subgraph induced by color c_prime? 
        # No, that's for trees. A cycle is formed if v connects two vertices in the same component 
        # of the subgraph induced by c_prime? 
        # Wait, the subgraph induced by c_prime might already have cycles? 
        # No, we maintain the invariant that the current partial coloring is valid.
        # So the subgraph induced by c_prime is a forest.
        # Adding v (colored c) to the mix of c and c_prime... 
        # Actually, we are checking the pair (c, c_prime).
        # The vertices involved are V_c (current) and V_cp (current).
        # v is being added to V_c.
        # So we check the subgraph induced by (V_c U {v}) and V_cp.
        # Since V_c is a forest and V_cp is a forest, and there are no edges between V_c and V_cp 
        # except those incident to v? No, there can be edges between V_c and V_cp.
        # The constraint is: The subgraph induced by ALL vertices with color c OR c_prime is a forest.
        
        # Let's do a BFS/DFS on the subgraph induced by colors {c, c_prime} including v.
        # To be fast, we only care about components that touch v.
        
        visited = set()
        stack = [v]
        visited.add(v)
        
        # We need to detect a cycle in the subgraph induced by {c, c_prime}.
        # Standard cycle detection in a graph.
        # Since the graph is small, we can just run BFS from v restricted to nodes with color c or c_prime.
        # If we encounter a visited node that is not the parent, it's a cycle.
        
        parent = {v: -1}
        while stack:
            u = stack.pop()
            for nb in adj[u]:
                if colors[nb] != c and colors[nb] != c_prime:
                    continue
                if nb == v and colors[nb] == c: # Self loop? No simple graph.
                    continue
                if nb not in visited:
                    visited.add(nb)
                    parent[nb] = u
                    stack.append(nb)
                else:
                    # Found a visited neighbor
                    # If it's the parent, it's just the tree edge
                    if parent[u] == nb:
                        continue
                    # Check if it's a back edge to an ancestor? 
                    # In undirected graph, if we see a visited node that is not parent, 
                    # it indicates a cycle IF the node is in the current DFS tree path.
                    # However, simple BFS cycle detection:
                    # If we find an edge (u, nb) where nb is visited and nb != parent[u], 
                    # does it always imply a cycle?
                    # Yes, in an undirected graph, any non-tree edge creates a cycle.
                    # But we must ensure we don't count the edge (u, parent[u]) as a cycle.
                    # The `if parent[u] == nb: continue` handles that.
                    # What if nb is a descendant? In BFS, we don't distinguish ancestor/descendant easily.
                    # But in an undirected graph, if (u, nb) is an edge and nb is visited, 
                    # and nb is not parent[u], then there is a cycle.
                    # Wait, if nb is a child of u, parent[nb] == u. 
                    # When we process u, we look at nb. If nb is already visited, it must have been 
                    # visited by someone else or is the parent.
                    # If nb is a child, it is NOT visited when we first see it from u? 
                    # No, we mark visited when pushing.
                    # So if we see nb visited, it means it was pushed earlier.
                    # If parent[nb] != u, then there is another path to nb. Cycle.
                    # If parent[nb] == u, then nb is a child, and we are looking at the tree edge. 
                    # But we check `parent[u] == nb`. That's the reverse edge.
                    # We need to check if `parent[nb] == u`? No, that's forward.
                    # In undirected graph, edge (u, nb) is seen from both sides.
                    # If we are at u, and nb is visited:
                    # 1. nb is parent of u -> skip.
                    # 2. nb is child of u -> skip? No, if nb is child, it was visited by u. 
                    #    But we only push unvisited nodes. So if nb is visited, it wasn't pushed by u 
                    #    in this step? 
                    #    Actually, when we process u, we iterate neighbors.
                    #    If nb is unvisited, we push it and set parent.
                    #    If nb is visited, it was pushed by someone else (or is parent).
                    #    So if nb is visited and nb != parent[u], it IS a cycle.
                    
                    # Let's double check.
                    # Graph: 1-2-3-1.
                    # Start 1. Push 1. Visited {1}.
                    # Pop 1. Neighbors 2, 3.
                    #   2 unvisited. Push 2. Parent[2]=1. Visited {1,2}.
                    #   3 unvisited. Push 3. Parent[3]=1. Visited {1,2,3}.
                    # Pop 3. Neighbors 1, 2.
                    #   1 visited. parent[3]=1. Skip.
                    #   2 visited. parent[3]=1 != 2. Cycle detected. Correct.
                    
                    return True
        return False

    def is_valid_assignment(v, c):
        # 1. Check proper coloring
        for nb in adj[v]:
            if colors[nb] == c:
                return False
        
        # 2. Check bicolored cycles with all other assigned colors
        assigned_colors = set(colors)
        assigned_colors.discard(-1)
        for c_prime in assigned_colors:
            if c_prime == c:
                continue
            if creates_cycle(v, c, c_prime):
                return False
        return True

    def backtrack(idx):
        if time.time() > deadline:
            return False
        
        if idx == n:
            return True
        
        v = order[idx]
        
        # Try colors 0, 1, 2
        # Heuristic: Try colors that appear less often? Or just 0,1,2
        for c in range(3):
            if is_valid_assignment(v, c):
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                colors[v] = -1
        return False

    # Run backtracking
    if backtrack(0):
        # Ensure Python ints
        return [int(c) for c in colors]
    
    # Fallback: Local Search if backtracking fails or times out
    # (Backtracking should work for N=50 if a solution exists)
    # If it times out, we might not have a solution. 
    # But the task implies a solution exists.
    # If backtracking times out, we return a greedy coloring as a last resort to avoid 0 score?
    # No, invalid result scores 0.
    # We must ensure we return something valid if possible.
    
    # If backtracking fails, try a simple local search with 4 colors as fallback
    # But 3 colors is the target.
    
    # Let's add a timeout guard for the backtracking itself.
    # The `time.time() > deadline` check is inside `backtrack`.
    # If it returns False due to timeout, colors might be partially assigned.
    # We need to detect if it was a timeout or a dead end.
    # Since `deadline` is 85s, and backtracking is exponential, it might hit deadline.
    
    # If backtracking fails, we fall back to a robust local search for 3 colors.
    # This is just in case.
    
    colors = [random.randint(0, 2) for _ in range(n)]
    
    def check_bicolored_forest_fast(colors, c1, c2):
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
                return False
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return True
        
        for u, v in edges:
            if (colors[u] == c1 or colors[u] == c2) and (colors[v] == c1 or colors[v] == c2):
                if not union(u, v):
                    return False
        return True

    def count_violations(colors):
        violations = 0
        for u, v in edges:
            if colors[u] == colors[v]:
                violations += 1
        for i in range(3):
            for j in range(i+1, 3):
                if not check_bicolored_forest_fast(colors, i, j):
                    violations += 1
        return violations

    current_violations = count_violations(colors)
    max_iter = 20000
    for i in range(max_iter):
        if time.time() > deadline:
            break
        if current_violations == 0:
            return [int(c) for c in colors]
        
        v = random.randint(0, n-1)
        old_c = colors[v]
        best_c = old_c
        best_v = current_violations
        
        for c in range(3):
            if c == old_c: continue
            # Quick proper check
            conflict = False
            for nb in adj[v]:
                if colors[nb] == c:
                    conflict = True
                    break
            if conflict: continue
            
            colors[v] = c
            v_new = count_violations(colors)
            if v_new < best_v:
                best_v = v_new
                best_c = c
            colors[v] = old_c
            
        colors[v] = best_c
        current_violations = best_v

    return [int(c) for c in colors]

# EVOLVE-BLOCK-END
