"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

import sys
sys.setrecursionlimit(10000)

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Degree of each vertex
    degree = [len(adj[i]) for i in range(n)]
    
    # Order vertices by descending degree, then by index for determinism
    order = sorted(range(n), key=lambda i: (-degree[i], i))
    
    # We'll use backtracking with forward checking.
    # State:
    #   colors[v] = color assigned to vertex v, or -1 if unassigned
    #   For each color pair (c1, c2) with c1 < c2, we maintain a Union-Find structure
    #   over the vertices that have been assigned colors c1 or c2.
    #   Actually, we need to track connected components in the subgraph induced by
    #   vertices of color c1 or c2. When we assign a new vertex v color c, we need to
    #   check for each neighbor u of v:
    #     - If colors[u] == c, reject (adjacent same color)
    #     - If colors[u] = c' != c, then we're adding edge (u,v) to the subgraph 
    #       induced by colors {c, c'}. We need to check if u and v are already 
    #       connected in that subgraph. If yes, reject.
    
    # Since the number of colors is small (we want to minimize), we can use a simpler
    # approach: for each color pair, maintain the set of connected components as a 
    # dictionary mapping vertex -> component_id, and for each component, the set of vertices.
    
    # Actually, let's use Union-Find per color pair. But since we're doing backtracking,
    # we need to be able to undo operations. Let's use a different approach:
    # For each color pair (c1, c2), maintain a list of edges in the subgraph, and 
    # check for cycles by doing BFS/DFS when needed. But that might be slow.
    
    # Better: maintain for each color pair a Union-Find with rollback. But that's complex.
    
    # Alternative: Since we're minimizing colors, let's try colors 1, 2, 3, 4, ... in order.
    # For each k, try to color with k colors using backtracking. If we find a solution, return it.
    
    # For the Union-Find with rollback, we can use a stack of operations.
    
    # Let's define: for each pair (i, j) with i < j, we have a Union-Find structure.
    # We'll use a dictionary: uf[(i,j)] = {parent: {}, rank: {}}
    # And a global stack for rollback.
    
    # Actually, let me think more carefully. The key constraint is:
    # When we assign vertex v color c, for each neighbor u with color c' != c:
    #   - The edge (u, v) is added to the subgraph induced by colors {c, c'}.
    #   - We must ensure that u and v are not already connected in that subgraph.
    
    # So for each color pair, we need to maintain connectivity. Let's use Union-Find
    # with path compression disabled (for easy rollback) or with a rollback stack.
    
    # Let's use a simpler approach: for each color pair, maintain a set of edges,
    # and when we add an edge, check if it creates a cycle by doing BFS. Since the
    # subgraphs are small (at most 50 vertices), this should be fast enough.
    
    # Actually, even simpler: maintain for each color pair a list of edges, and a 
    # connectivity structure. Let's use Union-Find without path compression, with 
    # a rollback stack.
    
    # Let me implement Union-Find with rollback for each color pair.
    
    # State for backtracking:
    # colors: list of length n, -1 for unassigned
    # For each color pair (i, j), a Union-Find structure with rollback capability.
    
    # Let's define a class for RollbackUF:
    
    class RollbackUF:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
            self.stack = []
        
        def find(self, x):
            while self.parent[x] != x:
                x = self.parent[x]
            return x
        
        def union(self, x, y):
            rx = self.find(x)
            ry = self.find(y)
            if rx == ry:
                return False  # already connected
            self.stack.append((rx, ry, self.rank[rx], self.rank[ry]))
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            return True
        
        def snapshot(self):
            return len(self.stack)
        
        def rollback(self, snap):
            while len(self.stack) > snap:
                rx, ry, rrx, rry = self.stack.pop()
                self.parent[ry] = ry
                self.rank[rx] = rrx
                self.rank[ry] = rry
    
    # We'll use a dictionary to store RollbackUF for each color pair.
    # Since the number of colors is small, this is manageable.
    
    def try_coloring(max_colors):
        colors = [-1] * n
        # uf_map: (c1, c2) -> RollbackUF
        uf_map = {}
        
        # Pre-allocate UF structures for all pairs up to max_colors
        for c1 in range(max_colors):
            for c2 in range(c1 + 1, max_colors):
                uf_map[(c1, c2)] = RollbackUF(n)
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = order[idx]
            
            # Try each color from 0 to max_colors-1
            for c in range(max_colors):
                # Check if any neighbor has color c
                conflict = False
                for u in adj[v]:
                    if colors[u] == c:
                        conflict = True
                        break
                if conflict:
                    continue
                
                # Take snapshots of all UF structures that might be affected
                # Only UFs for pairs (c, c') where c' is a color of a neighbor
                affected_pairs = set()
                for u in adj[v]:
                    if colors[u] != -1 and colors[u] != c:
                        c1 = min(c, colors[u])
                        c2 = max(c, colors[u])
                        affected_pairs.add((c1, c2))
                
                snapshots = {}
                for pair in affected_pairs:
                    snapshots[pair] = uf_map[pair].snapshot()
                
                # Check if adding v with color c creates a cycle in any affected pair
                valid = True
                for u in adj[v]:
                    if colors[u] != -1 and colors[u] != c:
                        c1 = min(c, colors[u])
                        c2 = max(c, colors[u])
                        pair = (c1, c2)
                        # Check if u and v are already connected in this pair's subgraph
                        uf = uf_map[pair]
                        if uf.find(u) == uf.find(v):
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Assign color c to v
                colors[v] = c
                
                # Union v with all neighbors that have a different color
                for u in adj[v]:
                    if colors[u] != -1 and colors[u] != c:
                        c1 = min(c, colors[u])
                        c2 = max(c, colors[u])
                        pair = (c1, c2)
                        uf_map[pair].union(u, v)
                
                if backtrack(idx + 1):
                    return True
                
                # Rollback
                colors[v] = -1
                for pair in affected_pairs:
                    uf_map[pair].rollback(snapshots[pair])
            
            return False
        
        return backtrack(0), colors
    
    # Try with increasing number of colors
    for k in range(1, n + 1):
        success, colors = try_coloring(k)
        if success:
            # Verify the solution
            # Check adjacent vertices have different colors
            for u, v in edges:
                if colors[u] == colors[v]:
                    raise ValueError("Invalid coloring: adjacent vertices have same color")
            
            # Check no bicolored cycles
            # For each color pair, check that the induced subgraph is a forest
            from collections import defaultdict
            for c1 in range(k):
                for c2 in range(c1 + 1, k):
                    # Collect edges in the subgraph induced by colors c1 and c2
                    subgraph_edges = []
                    for u, v in edges:
                        cu, cv = colors[u], colors[v]
                        if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                            subgraph_edges.append((u, v))
                    
                    # Check for cycles using Union-Find
                    if not subgraph_edges:
                        continue
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
                    
                    has_cycle = False
                    for u, v in subgraph_edges:
                        if not union(u, v):
                            has_cycle = True
                            break
                    
                    if has_cycle:
                        raise ValueError("Invalid coloring: bicolored cycle detected")
            
            return [int(c) for c in colors]
    
    # Should not reach here
    return list(range(n))

# EVOLVE-BLOCK-END
