"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    deadline = time.time() + 55.0
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute: for each pair of colors, which edges are relevant
    # We'll use a fast bicolored cycle check with arrays instead of dicts
    
    def has_bicolored_cycle_fast(colors, c1, c2):
        """Fast check: does the subgraph induced by colors c1 and c2 have a cycle?"""
        # Count vertices and edges in the bicolored subgraph
        # A graph is a forest iff E = V - components
        # Faster: just do BFS/DFS cycle detection on the induced subgraph
        # Use a simple approach: count edges where both endpoints have c1 or c2
        # and check if that count >= number of such vertices (cycle exists iff E >= V for connected, 
        # but more generally: cycle exists iff E >= V for the whole subgraph)
        
        vcount = 0
        ecount = 0
        for i, c in enumerate(colors):
            if c == c1 or c == c2:
                vcount += 1
        if vcount == 0:
            return False
        for u, v in edges:
            cu, cv = colors[u], colors[v]
            if (cu == c1 or cu == c2) and (cv == c1 or cv == c2):
                ecount += 1
        # A forest has at most V-1 edges. If E >= V, there's a cycle.
        # Actually, a forest on V vertices has exactly V - C edges where C is number of components.
        # So E <= V - 1 always for a forest. If E >= V, definitely a cycle.
        # But E < V doesn't guarantee no cycle (e.g., two components, one with a cycle).
        # We need the exact check. Use union-find with arrays.
        
        # Map vertices to indices for union-find
        parent = [-1] * n
        rank = [0] * n
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        for i, c in enumerate(colors):
            if c == c1 or c == c2:
                parent[i] = i
        
        for u, v in edges:
            cu, cv = colors[u], colors[v]
            if (cu == c1 or cu == c2) and (cv == c1 or cv == c2):
                pu, pv = find(u), find(v)
                if pu == pv:
                    return True
                if rank[pu] < rank[pv]:
                    pu, pv = pv, pu
                parent[pv] = pu
                if rank[pu] == rank[pv]:
                    rank[pu] += 1
        return False
    
    def is_valid(colors):
        """Check if coloring is fully valid."""
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        used = set(colors)
        for c1, c2 in combinations(used, 2):
            if has_bicolored_cycle_fast(colors, c1, c2):
                return False
        return True
    
    # Use a smarter ordering: process vertices in an order that maximizes constraints
    # For a grid-like graph, processing in row-major order works well
    # Let's try natural order first, but with forward-checking
    
    for k in range(2, 12):
        if time.time() > deadline:
            break
        
        colors = [-1] * n
        # Track which colors are used
        color_used = [False] * k
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return is_valid(colors)
            
            v = idx
            # Try each color, ordered by least-constraining first
            # Simple: try colors that are already used first (to minimize distinct colors)
            for c in range(k):
                # Check adjacent constraint
                valid = True
                for u in adj[v]:
                    if colors[u] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                colors[v] = c
                color_used[c] = True
                
                # Early check: for each other used color, check no bicolored cycle
                ok = True
                for other_c in range(k):
                    if other_c != c and color_used[other_c]:
                        if has_bicolored_cycle_fast(colors, c, other_c):
                            ok = False
                            break
                
                if ok:
                    if backtrack(idx + 1):
                        return True
                
                colors[v] = -1
                color_used[c] = False
            
            return False
        
        if backtrack(0):
            result = [int(c) for c in colors]
            return result
    
    # Fallback: use all distinct colors
    return [int(i) for i in range(n)]

# EVOLVE-BLOCK-END
