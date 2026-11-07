"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
sys.setrecursionlimit(10000)

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Order vertices by degree (all are 4, so just use natural order or a heuristic)
    # Use a simple ordering: vertices with more constrained neighbors first
    # Since all have degree 4, use a static ordering based on position
    order = list(range(n))
    
    # Backtracking search
    colors = [-1] * n
    
    def find(parent, x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(parent, rank, a, b):
        ra, rb = find(parent, a), find(parent, b)
        if ra == rb:
            return False  # cycle detected
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True
    
    # For each pair of colors, maintain a union-find structure
    # Key: (color1, color2) where color1 < color2
    # Value: (parent_dict, rank_dict, edge_count)
    
    def backtrack(idx):
        if time.time() > deadline:
            return None
        if idx == n:
            return list(colors)
        
        v = order[idx]
        # Get available colors (not used by neighbors)
        used_by_neighbors = set()
        for nb in adj[v]:
            if colors[nb] >= 0:
                used_by_neighbors.add(colors[nb])
        
        for c in range(50):
            if c in used_by_neighbors:
                continue
            
            # Try assigning color c to vertex v
            colors[v] = c
            
            # Check bicolored cycle constraint
            # For each neighbor with a different color, we need to check
            # if adding this edge creates a cycle in the bicolored subgraph
            valid = True
            new_unions = []  # (key, a, b) for rollback
            
            for nb in adj[v]:
                nc = colors[nb]
                if nc == c:
                    continue  # same color, not a bicolored edge
                # This is a bicolored edge between colors c and nc
                c1, c2 = min(c, nc), max(c, nc)
                key = (c1, c2)
                
                # Get or create the union-find for this color pair
                if key not in uf_data:
                    uf_data[key] = {'parent': {}, 'rank': {}}
                uf = uf_data[key]
                parent_dict = uf['parent']
                rank_dict = uf['rank']
                
                # Initialize if needed
                if c1 not in parent_dict:
                    parent_dict[c1] = c1
                    rank_dict[c1] = 0
                if c2 not in parent_dict:
                    parent_dict[c2] = c2
                    rank_dict[c2] = 0
                
                # We need to track which vertex belongs to which "node" in the UF
                # Actually, the UF should be over vertices, not colors
                # Let me reconsider: for a bicolored subgraph (colors c1, c2),
                # we have vertices colored c1 or c2, and edges between them.
                # The UF should be over the vertices in this subgraph.
                
                # Actually, I need to redo this. The bicolored subgraph for colors (c1, c2)
                # consists of all vertices with color c1 or c2, and edges between them.
                # A cycle in this subgraph is what we want to avoid.
                # So the UF should be over vertices, and we union endpoints of edges
                # that are both in {c1, c2}.
                
                # Let me restructure: for each color pair, maintain a UF over vertices
                pass
            
            # This approach is getting complex. Let me use a different strategy.
            colors[v] = -1
            break
        
        return None
    
    # Simpler approach: for each candidate coloring, check validity
    # Use backtracking with incremental checking
    
    # For the bicolored cycle check, I'll maintain for each color pair a set of edges
    # and check if adding a new edge creates a cycle using UF
    
    uf_data = {}  # key: (c1, c2) -> {'parent': {vertex: parent}, 'rank': {vertex: rank}}
    
    def get_uf(c1, c2):
        key = (min(c1, c2), max(c1, c2))
        if key not in uf_data:
            uf_data[key] = {'parent': {}, 'rank': {}}
        return uf_data[key]
    
    def uf_find(uf, x):
        parent = uf['parent']
        if x not in parent:
            parent[x] = x
            uf['rank'][x] = 0
            return x
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def uf_union(uf, a, b):
        parent = uf['parent']
        rank = uf['rank']
        ra = uf_find(uf, a)
        rb = uf_find(uf, b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True
    
    def backtrack(idx):
        if time.time() > deadline:
            return None
        if idx == n:
            return list(colors)
        
        v = order[idx]
        used_by_neighbors = set()
        for nb in adj[v]:
            if colors[nb] >= 0:
                used_by_neighbors.add(colors[nb])
        
        # Try colors in order, prefer smaller colors
        for c in range(50):
            if c in used_by_neighbors:
                continue
            
            colors[v] = c
            
            # Check if adding v with color c creates a bicolored cycle
            # For each neighbor nb with color nc != c, the edge (v, nb) is in the
            # bicolored subgraph for colors (c, nc)
            valid = True
            new_unions = []  # (key, uf, a, b) for potential rollback (but we don't rollback UF easily)
            
            # Since we're doing backtracking, we need to be able to undo unions.
            # Alternative: don't maintain UF incrementally, just check at the end.
            # But that's expensive.
            
            # For now, let's just do a full check if we complete the coloring.
            # Actually, let's try a different approach: use a greedy method with
            # local search or just try random assignments and check.
            
            colors[v] = -1
            break
        
        return None
    
    # Given the complexity, let me try a simpler approach:
    # 1. Try to find a valid coloring with k colors using backtracking
    # 2. For the bicolored cycle check, use a simple O(E) check per candidate
    
    def is_valid(coloring):
        # Check adjacent vertices have different colors
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        
        # Check no bicolored cycles
        # For each pair of colors, build the subgraph and check for cycles
        from collections import defaultdict
        color_pairs = defaultdict(list)
        for u, v in edges:
            cu, cv = coloring[u], coloring[v]
            if cu != cv:
                c1, c2 = min(cu, cv), max(cu, cv)
                color_pairs[(c1, c2)].append((u, v))
        
        for (c1, c2), edge_list in color_pairs.items():
            # Check if this subgraph has a cycle using BFS/DFS
            # Build adjacency for vertices with color c1 or c2 that have edges in this pair
            parent = {}
            def find(x):
                if x not in parent:
                    parent[x] = x
                    return x
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]
            
            for u, v in edge_list:
                ru, rv = find(u), find(v)
                if ru == rv:
                    return False  # cycle
                parent[ru] = rv
        
        return True
    
    # Try increasing k
    for k in range(2, 50):
        if time.time() > deadline:
            break
        
        # Backtracking with k colors
        colors = [-1] * n
        
        def bt(idx):
            if time.time() > deadline:
                return None
            if idx == n:
                return list(colors)
            
            v = order[idx]
            used = set()
            for nb in adj[v]:
                if colors[nb] >= 0:
                    used.add(colors[nb])
            
            for c in range(k):
                if c in used:
                    continue
                
                colors[v] = c
                
                # Quick check: for each neighbor with different color, 
                # check if this creates a bicolored cycle
                # This is expensive, so let's just do it at the end for now
                # But we can do a quick check: if the subgraph for any color pair
                # involving c already has enough edges to form a cycle...
                
                result = bt(idx + 1)
                if result is not None:
                    return result
                
                colors[v] = -1
            
            return None
        
        result = bt(0)
        if result is not None:
            # Verify
            if is_valid(result):
                return [int(x) for x in result]
            else:
                # Should not happen if is_valid is correct
                continue
    
    # Fallback: return a valid coloring (use all different colors)
    return [int(i) for i in range(n)]
# EVOLVE-BLOCK-END
