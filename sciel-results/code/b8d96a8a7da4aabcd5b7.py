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
    
    # Precompute edge list indexed for quick lookup
    edge_set = set()
    for u, v in edges:
        if u < v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))
    
    # Try with 3 colors first, then 4, etc.
    for num_colors in [3, 4, 5, 6]:
        assignment = [-1] * n
        domains = [list(range(num_colors)) for _ in range(n)]
        
        # For incremental bicolored cycle detection:
        # For each color pair (c1, c2), maintain a count of active edges
        # and use a union-find that we rebuild only when needed
        # Key optimization: only check pairs that could be affected
        
        def check_bicolored_cycle(assignment, num_colors, changed_vertex=-1, changed_color=-1):
            """Check if any pair of colors forms a cycle. If changed_vertex is given,
            only check pairs involving changed_color."""
            if changed_vertex >= 0 and changed_color >= 0:
                # Only check pairs involving the changed color
                pairs_to_check = [(changed_color, c) for c in range(num_colors) if c != changed_color]
            else:
                pairs_to_check = [(c1, c2) for c1 in range(num_colors) for c2 in range(c1 + 1, num_colors)]
            
            for c1, c2 in pairs_to_check:
                # Check if this color pair has a cycle using union-find
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
                
                # Only process edges where both endpoints have colors in {c1, c2}
                for u, v in edges:
                    au = assignment[u]
                    av = assignment[v]
                    if au == -1 or av == -1:
                        continue
                    if au in (c1, c2) and av in (c1, c2):
                        if not union(u, v):
                            return True
            return False
        
        def backtrack(depth):
            if depth == n:
                return True
            
            # MRV with degree tie-breaking
            min_domain_size = float('inf')
            min_vertex = -1
            min_degree = float('inf')
            for i in range(n):
                if assignment[i] == -1:
                    ds = len(domains[i])
                    deg = len(adj[i])
                    if ds < min_domain_size or (ds == min_domain_size and deg > min_degree):
                        min_domain_size = ds
                        min_vertex = i
                        min_degree = deg
                        if ds == 0:
                            return False
            
            if min_vertex == -1:
                return True
            
            # Order colors by least constraining value (fewest conflicts with neighbors)
            color_order = []
            for c in domains[min_vertex]:
                conflict_count = 0
                for nb in adj[min_vertex]:
                    if assignment[nb] == c:
                        conflict_count += 1
                color_order.append((conflict_count, c))
            color_order.sort()
            
            for _, c in color_order:
                assignment[min_vertex] = c
                
                # Forward checking
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
                    for nb, d in saved_domains.items():
                        domains[nb] = d
                    assignment[min_vertex] = -1
                    continue
                
                # Check bicolored cycle - only for pairs involving color c
                if check_bicolored_cycle(assignment, num_colors, min_vertex, c):
                    for nb, d in saved_domains.items():
                        domains[nb] = d
                    assignment[min_vertex] = -1
                    continue
                
                if backtrack(depth + 1):
                    return True
                
                for nb, d in saved_domains.items():
                    domains[nb] = d
                assignment[min_vertex] = -1
            
            return False
        
        # Symmetry breaking: fix vertex 0 to color 0
        assignment[0] = 0
        domains[0] = [0]
        for nb in adj[0]:
            if 0 in domains[nb]:
                domains[nb] = [c for c in domains[nb] if c != 0]
        
        if backtrack(1):
            result = list(assignment)
            for i in range(n):
                assert isinstance(result[i], int)
                assert 0 <= result[i] < 50
            return result
    
    # Fallback: use all distinct colors
    return list(range(n))
# EVOLVE-BLOCK-END
