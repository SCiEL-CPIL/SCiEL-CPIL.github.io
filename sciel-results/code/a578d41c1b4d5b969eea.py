"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Deadline guard: 85 seconds (5s margin before 90s kill)
    deadline = time.time() + 85.0
    
    # We want to minimize the number of distinct colors.
    # Try k = 2, 3, 4, ... and return the first valid coloring found.
    # The graph is degree 4, so 4-coloring is always possible (greedy),
    # but we need the "bicolored cycle" constraint: for any two colors,
    # the subgraph induced by vertices of those two colors must be a forest.
    
    # A bicolored cycle is a cycle where all vertices have only two colors.
    # So we need: for every pair of colors (c1, c2), the subgraph induced by
    # vertices colored c1 or c2 is acyclic.
    
    def is_forest(coloring, color_set):
        """Check if subgraph induced by vertices with colors in color_set is a forest."""
        # Build the subgraph
        visited = set()
        
        def has_cycle(start):
            stack = [(start, -1)]
            while stack:
                node, parent = stack.pop()
                if node in visited:
                    return True
                visited.add(node)
                for neighbor in adj[node]:
                    if coloring[neighbor] in color_set:
                        if neighbor == parent:
                            continue
                        if neighbor in visited:
                            return True
                        stack.append((neighbor, node))
            return False
        
        for v in range(n):
            if coloring[v] in color_set and v not in visited:
                if has_cycle(v):
                    return False
        return True
    
    def check_no_bicolored_cycle(coloring, k):
        """Check that for every pair of colors, the induced subgraph is a forest."""
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                if not is_forest(coloring, {c1, c2}):
                    return False
        return True
    
    # Backtracking search with MRV heuristic
    def solve(k):
        """Try to find a valid k-coloring using backtracking with MRV."""
        coloring = [-1] * n
        # Domains: initially all k colors for each vertex
        domains = [list(range(k)) for _ in range(n)]
        
        # Order vertices by degree (highest first) for better pruning
        order = sorted(range(n), key=lambda v: -len(adj[v]))
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == n:
                return check_no_bicolored_cycle(coloring, k)
            
            v = order[idx]
            
            # Try each color in the domain
            for c in domains[v][:]:
                # Check adjacency constraint
                valid = True
                for u in adj[v]:
                    if coloring[u] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Assign
                coloring[v] = c
                domains[v] = [c]
                
                # Forward checking: update domains of unassigned neighbors
                saved_domains = []
                fc_valid = True
                for u in adj[v]:
                    if coloring[u] == -1:
                        saved_domains.append((u, domains[u][:]))
                        if c in domains[u]:
                            domains[u] = [x for x in domains[u] if x != c]
                            if not domains[u]:
                                fc_valid = False
                                break
                
                if fc_valid:
                    if backtrack(idx + 1):
                        return True
                else:
                    # Restore domains
                    pass
                
                # Restore domains
                for u, old_dom in saved_domains:
                    domains[u] = old_dom
                domains[v] = list(range(k))
                coloring[v] = -1
            
            return False
        
        return backtrack(0)
    
    # Try increasing k
    best_coloring = None
    for k in range(2, 5):
        if time.time() > deadline:
            break
        if solve(k):
            best_coloring = coloring[:]
            break
    
    if best_coloring is None:
        # Fallback: use 4 colors with a simple greedy that avoids bicolored cycles
        # Actually, let's try a random restart approach
        import random
        random.seed(42)
        for attempt in range(10000):
            if time.time() > deadline:
                break
            coloring = [random.randint(0, 3) for _ in range(n)]
            # Fix adjacency conflicts
            valid = True
            for u, v in edges:
                if coloring[u] == coloring[v]:
                    valid = False
                    break
            if not valid:
                continue
            if check_no_bicolored_cycle(coloring, 4):
                best_coloring = coloring
                break
    
    if best_coloring is None:
        # Last resort: assign unique colors (always valid)
        best_coloring = list(range(n))
    
    # Ensure all values are Python ints
    return [int(c) for c in best_coloring]
# EVOLVE-BLOCK-END
