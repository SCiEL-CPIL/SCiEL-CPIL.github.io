"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Order vertices by degree (descending) for variable ordering
    deg = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: -deg[x])
    # Map vertex index to position in ordering
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i
    
    # For each vertex, precompute neighbors that come before it in the ordering
    # (i.e., already assigned when we process this vertex)
    prev_neighbors = [[] for _ in range(n)]
    for v in range(n):
        for u in adj[v]:
            if pos[u] < pos[v]:
                prev_neighbors[v].append(u)
    
    # We'll try to find a coloring with k colors, for k = 2, 3, 4, ...
    # The constraint: no bicolored cycle. This means for any pair of colors (c1, c2),
    # the subgraph induced by vertices with colors c1 or c2 must be a forest.
    #
    # Key insight: A bicolored cycle exists if and only if there's a cycle where 
    # vertices alternate between two colors. More precisely, for any two colors c1, c2,
    # consider the graph G_{c1,c2} whose vertices are those colored c1 or c2, and edges 
    # are original edges between such vertices. This must be a forest.
    #
    # A forest on m vertices has at most m-1 edges. So for each color pair, the number of 
    # edges in the induced subgraph must be <= (number of vertices in that subgraph) - 1.
    #
    # For backtracking with forward checking, we maintain for each color pair the number 
    # of edges and vertices, and check the forest constraint incrementally.
    #
    # Actually, a simpler check: when we assign a color to vertex v, for each neighbor u 
    # already assigned, we need to check that adding the edge (u,v) doesn't create a 
    # bicolored cycle. An edge (u,v) creates a bicolored cycle with colors (cu, cv) if 
    # and only if u and v are already connected in the subgraph induced by colors cu and cv.
    
    def solve_with_k(k):
        """Try to find a valid k-coloring. Returns list of colors or None."""
        color = [-1] * n  # -1 means unassigned
        
        # For each color pair, track connectivity using union-find
        # color_pair_dsu[(c1,c2)] = DSU for the subgraph induced by colors c1 and c2
        # We'll use a dictionary mapping (c1,c2) to a DSU structure
        
        class DSU:
            def __init__(self, n):
                self.parent = list(range(n))
                self.rank = [0] * n
            
            def find(self, x):
                while self.parent[x] != x:
                    self.parent[x] = self.parent[self.parent[x]]
                    x = self.parent[x]
                return x
            
            def union(self, x, y):
                rx, ry = self.find(x), self.find(y)
                if rx == ry:
                    return False  # Already connected, would create cycle
                if self.rank[rx] < self.rank[ry]:
                    rx, ry = ry, rx
                self.parent[ry] = rx
                if self.rank[rx] == self.rank[ry]:
                    self.rank[rx] += 1
                return True  # Successfully merged
        
        # dsu[(c1,c2)] where c1 < c2, maps to DSU object
        # The DSU indices are vertex indices (0..n-1)
        dsu = {}
        
        def get_dsu(c1, c2):
            key = (min(c1,c2), max(c1,c2))
            if key not in dsu:
                dsu[key] = DSU(n)
            return dsu[key]
        
        def backtrack(idx):
            """Assign colors to vertices in order[idx:]."""
            if idx == n:
                return True
            
            v = order[idx]
            
            # Try each color
            for c in range(k):
                # Check adjacency constraint: no neighbor with same color
                valid = True
                for u in prev_neighbors[v]:
                    if color[u] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check bicolored cycle constraint
                # For each neighbor u with color cu != c, check that u and v are not 
                # already connected in the subgraph induced by colors (cu, c)
                valid = True
                new_dsu_entries = []  # (key, dsu, u, v) for rollback
                for u in prev_neighbors[v]:
                    cu = color[u]
                    if cu == c:
                        continue
                    key = (min(cu, c), max(cu, c))
                    if key in dsu:
                        d = dsu[key]
                        ru, rv = d.find(u), d.find(v)
                        if ru == rv:
                            valid = False
                            break
                        # Will need to union
                        new_dsu_entries.append((key, d, u, v))
                    # If key not in dsu, this is the first edge for this color pair,
                    # so no cycle possible
                if not valid:
                    # Rollback any unions we did (but we didn't actually do them yet)
                    continue
                
                # Assign color
                color[v] = c
                
                # Apply unions
                for key, d, u, vv in new_dsu_entries:
                    d.union(u, vv)
                
                if backtrack(idx + 1):
                    return True
                
                # Rollback
                color[v] = -1
                for key, d, u, vv in new_dsu_entries:
                    # We can't easily rollback a union, so we need a different approach
                    # Let's use a different strategy: instead of unioning, just check
                    # connectivity before committing
                    pass
                
                return False
            
            return False
        
        return None  # This simple approach doesn't work well with rollback
    
    # Better approach: use a different strategy. 
    # Instead of backtracking with DSU rollback (which is complex), let's use a 
    # simpler heuristic-based approach.
    
    # For a 4-regular graph on 50 vertices, the chromatic number is at most 5 (by Brooks' theorem,
    # since the graph is not complete and not an odd cycle). But we need the additional 
    # bicolored forest constraint.
    
    # Let's try a greedy approach with local search.
    
    def greedy_color(k):
        """Greedy coloring with k colors, checking bicolored constraint."""
        color = [-1] * n
        
        # DSU for each color pair
        class DSU:
            def __init__(self, n):
                self.parent = list(range(n))
                self.rank = [0] * n
            
            def find(self, x):
                while self.parent[x] != x:
                    self.parent[x] = self.parent[self.parent[x]]
                    x = self.parent[x]
                return x
            
            def union(self, x, y):
                rx, ry = self.find(x), self.find(y)
                if rx == ry:
                    return False
                if self.rank[rx] < self.rank[ry]:
                    rx, ry = ry, rx
                self.parent[ry] = rx
                if self.rank[rx] == self.rank[ry]:
                    self.rank[rx] += 1
                return True
        
        dsu = {}
        
        def get_dsu(c1, c2):
            key = (min(c1,c2), max(c1,c2))
            if key not in dsu:
                dsu[key] = DSU(n)
            return dsu[key]
        
        for v in order:
            # Find valid colors
            valid_colors = []
            for c in range(k):
                # Check adjacency
                ok = True
                for u in adj[v]:
                    if color[u] == c:
                        ok = False
                        break
                if not ok:
                    continue
                
                # Check bicolored cycle
                ok = True
                for u in adj[v]:
                    if color[u] == -1:
                        continue
                    cu = color[u]
                    if cu == c:
                        continue
                    key = (min(cu, c), max(cu, c))
                    if key in dsu:
                        d = dsu[key]
                        if d.find(u) == d.find(v):
                            ok = False
                            break
                if ok:
                    valid_colors.append(c)
            
            if not valid_colors:
                return None
            
            # Pick the color that causes least future conflict
            # Simple heuristic: pick the first valid color
            c = valid_colors[0]
            color[v] = c
            
            # Update DSUs
            for u in adj[v]:
                if color[u] == -1:
                    continue
                cu = color[u]
                if cu == c:
                    continue
                key = (min(cu, c), max(cu, c))
                d = get_dsu(cu, c)
                d.union(u, v)
        
        return color
    
    # Try increasing k
    for k in range(2, 10):
        result = greedy_color(k)
        if result is not None:
            # Verify the solution
            if verify(result):
                return [int(x) for x in result]
    
    # If greedy fails, try a more sophisticated approach
    # Use simulated annealing or local search
    
    def verify(color):
        """Verify that a coloring is valid."""
        # Check adjacency
        for u, v in edges:
            if color[u] == color[v]:
                return False
        
        # Check bicolored cycles using union-find
        n = len(color)
        # For each pair of colors, check if the induced subgraph is a forest
        colors_used = set(color)
        for i, c1 in enumerate(colors_used):
            for c2 in list(colors_used)[i+1:]:
                # Build subgraph induced by colors c1 and c2
                # Use union-find to check for cycles
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
                    if color[u] in (c1, c2) and color[v] in (c1, c2):
                        if not union(u, v):
                            return False  # Cycle found
        
        return True
    
    # Fallback: use a simple heuristic that we know works
    # For a 4-regular graph, 4 colors should suffice with the right assignment
    # Let's try a pattern-based approach
    
    # Actually, let's just return a valid solution using a known method
    # The graph is 4-regular on 50 vertices. By Brooks' theorem, chi <= 4 
    # (since it's not K5 or an odd cycle). But we need the bicolored forest constraint.
    
    # Let's try a specific construction
    # Assign colors based on vertex index modulo some number
    
    # For now, return a simple valid solution using more colors if needed
    for k in range(5, 50):
        result = greedy_color(k)
        if result is not None and verify(result):
            return [int(x) for x in result]
    
    # Last resort: use all distinct colors
    return list(range(n))

# EVOLVE-BLOCK-END
