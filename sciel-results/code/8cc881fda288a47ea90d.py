"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Try with 3 colors first, then 4, etc.
    for num_colors in [3, 4, 5, 6]:
        colors = list(range(num_colors))
        assignment = [-1] * n
        
        # Precompute which color pairs would create a bicolored cycle
        # We'll use union-find during search to track bicolored connectivity
        
        def has_bicolored_cycle(assignment, num_colors):
            """Check if any pair of colors forms a cycle in the induced subgraph."""
            # For each pair of colors, build the induced subgraph and check for cycles
            for c1 in range(num_colors):
                for c2 in range(c1 + 1, num_colors):
                    # Build induced subgraph on vertices with color c1 or c2
                    # Use union-find to detect cycles
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
                    
                    for u in range(n):
                        if assignment[u] in (c1, c2):
                            for v in adj[u]:
                                if v > u and assignment[v] in (c1, c2):
                                    if not union(u, v):
                                        return True
            return False
        
        def backtrack(idx, domains, color_count):
            if idx == n:
                return True
            
            # MRV: pick unassigned vertex with smallest domain
            min_domain_size = float('inf')
            min_vertex = -1
            for i in range(n):
                if assignment[i] == -1:
                    if len(domains[i]) < min_domain_size:
                        min_domain_size = len(domains[i])
                        min_vertex = i
                        if min_domain_size == 0:
                            return False
            
            if min_vertex == -1:
                return True
            
            for c in list(domains[min_vertex]):
                # Try assigning color c to min_vertex
                assignment[min_vertex] = c
                
                # Forward checking: update domains of unassigned neighbors
                valid = True
                saved_domains = {}
                for nb in adj[min_vertex]:
                    if assignment[nb] == -1:
                        saved_domains[nb] = domains[nb][:]
                        if c in domains[nb]:
                            domains[nb] = [x for x in domains[nb] if x != c]
                            if len(domains[nb]) == 0:
                                valid = False
                                break
                
                if not valid:
                    # Restore
                    for nb, d in saved_domains.items():
                        domains[nb] = d
                    assignment[min_vertex] = -1
                    continue
                
                # Check bicolored cycle constraint
                if has_bicolored_cycle(assignment, num_colors):
                    for nb, d in saved_domains.items():
                        domains[nb] = d
                    assignment[min_vertex] = -1
                    continue
                
                if backtrack(idx + 1, domains, color_count + 1):
                    return True
                
                # Undo
                for nb, d in saved_domains.items():
                    domains[nb] = d
                assignment[min_vertex] = -1
            
            return False
        
        # Initialize domains
        domains = [list(colors) for _ in range(n)]
        
        # Symmetry breaking: fix first vertex to color 0
        assignment[0] = 0
        domains[0] = [0]
        for nb in adj[0]:
            domains[nb] = [c for c in domains[nb] if c != 0]
        
        if backtrack(0, domains, 1):
            # Verify solution
            used_colors = set(assignment)
            result = list(assignment)
            for i in range(n):
                assert isinstance(result[i], int)
                assert 0 <= result[i] < 50
            return result
    
    # Fallback: use all distinct colors
    return list(range(n))
# EVOLVE-BLOCK-END
