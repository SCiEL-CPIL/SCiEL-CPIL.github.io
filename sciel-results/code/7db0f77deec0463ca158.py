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
        domains = [list(colors) for _ in range(n)]
        
        # Union-Find structures for each pair of colors
        # Key: (c1, c2) with c1 < c2
        uf_parent = {}
        uf_rank = {}
        
        def init_uf(c1, c2):
            key = (min(c1, c2), max(c1, c2))
            if key not in uf_parent:
                uf_parent[key] = list(range(n))
                uf_rank[key] = [0] * n
        
        # Initialize all pairs
        for i in range(num_colors):
            for j in range(i + 1, num_colors):
                init_uf(i, j)
        
        def find(key, x):
            parent = uf_parent[key]
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(key, x, y):
            parent = uf_parent[key]
            rank = uf_rank[key]
            rx = find(key, x)
            ry = find(key, y)
            if rx == ry:
                return False # Cycle
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return True
        
        def backtrack():
            # MRV Heuristic
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
            
            v = min_vertex
            for c in list(domains[v]):
                # Try assigning color c to v
                assignment[v] = c
                
                # Snapshot UFs for pairs involving c
                affected_keys = []
                snapshots = {}
                for c2 in range(num_colors):
                    if c2 != c:
                        key = (min(c, c2), max(c, c2))
                        affected_keys.append(key)
                        snapshots[key] = (uf_parent[key][:], uf_rank[key][:])
                        
                # Check neighbors and update UFs
                valid = True
                for nb in adj[v]:
                    if assignment[nb] != -1:
                        c_nb = assignment[nb]
                        if c_nb != c:
                            key = (min(c, c_nb), max(c, c_nb))
                            if not union(key, v, nb):
                                valid = False
                                break
                    else:
                        # Forward checking: remove c from domain of unassigned neighbor
                        if c in domains[nb]:
                            domains[nb] = [x for x in domains[nb] if x != c]
                            if len(domains[nb]) == 0:
                                valid = False
                                break
                
                if not valid:
                    # Restore UFs
                    for key in affected_keys:
                        uf_parent[key], uf_rank[key] = snapshots[key]
                    # Restore domains
                    # We need to know which domains were changed. 
                    # Simpler: just recompute domains? No, that's expensive.
                    # We should track domain changes too.
                    # Let's restart domain tracking properly.
                    pass # Handled below
                else:
                    if backtrack():
                        return True
                    # Restore UFs on backtrack
                    for key in affected_keys:
                        uf_parent[key], uf_rank[key] = snapshots[key]
                
                # Restore assignment
                assignment[v] = -1
                
                # Restore domains of neighbors
                # We need to track which domains were modified.
                # Let's modify the loop to store original domains.
                
            return False
        
        # The above backtrack has a flaw in domain restoration.
        # Let's rewrite backtrack with proper domain tracking.
        
        def backtrack_v2():
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
            
            v = min_vertex
            for c in list(domains[v]):
                assignment[v] = c
                
                # Snapshot UFs
                affected_keys = []
                uf_snapshots = {}
                for c2 in range(num_colors):
                    if c2 != c:
                        key = (min(c, c2), max(c, c2))
                        affected_keys.append(key)
                        uf_snapshots[key] = (uf_parent[key][:], uf_rank[key][:])
                        
                # Domain changes
                domain_changes = {}
                valid = True
                for nb in adj[v]:
                    if assignment[nb] != -1:
                        c_nb = assignment[nb]
                        if c_nb != c:
                            key = (min(c, c_nb), max(c, c_nb))
                            if not union(key, v, nb):
                                valid = False
                                break
                    else:
                        if c in domains[nb]:
                            domain_changes[nb] = domains[nb][:]
                            domains[nb] = [x for x in domains[nb] if x != c]
                            if len(domains[nb]) == 0:
                                valid = False
                                break
                
                if valid:
                    if backtrack_v2():
                        return True
                
                # Restore
                for key in affected_keys:
                    uf_parent[key], uf_rank[key] = uf_snapshots[key]
                for nb, orig_dom in domain_changes.items():
                    domains[nb] = orig_dom
                assignment[v] = -1
            
            return False
        
        # Symmetry breaking: fix vertex 0 to color 0
        assignment[0] = 0
        domains[0] = [0]
        for nb in adj[0]:
            domains[nb] = [c for c in domains[nb] if c != 0]
            
        # Initialize UFs with the first assignment?
        # No, the backtrack function handles the unions as it goes.
        # But we already assigned vertex 0. We should process its unions now.
        # Actually, it's safer to let the backtrack function handle everything.
        # But we fixed vertex 0. So we should do the union steps for vertex 0 manually
        # or start backtrack with vertex 0 already assigned and UFs updated.
        
        # Let's update UFs for vertex 0's neighbors if they were assigned? No, they are unassigned.
        # So no unions yet.
        
        if backtrack_v2():
            used_colors = set(assignment)
            result = list(assignment)
            for i in range(n):
                assert isinstance(result[i], int)
                assert 0 <= result[i] < 50
            return result
    
    # Fallback: use all distinct colors
    return list(range(n))
# EVOLVE-BLOCK-END
