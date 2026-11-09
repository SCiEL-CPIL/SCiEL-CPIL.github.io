"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def construct():
    """Find a valid coloring with minimum number of distinct colors.
    
    Constraints:
    1. Adjacent vertices must have different colors.
    2. The subgraph induced by any two colors must be a forest (no bicolored cycles).
    
    Strategy: Backtracking search with incremental bicolored-cycle detection.
    Try colorings with increasing number of colors (3, 4, 5, ...) until valid.
    """
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Deadline guard
    deadline = time.time() + 80.0
    
    # For each color pair, track the union-find structure to detect cycles
    # We'll use a dictionary: (c1, c2) -> set of edges in that bicolored subgraph
    # To check if adding an edge creates a cycle, we need to know if the two 
    # endpoints are already connected in the bicolored subgraph.
    
    # More efficient: maintain for each color pair a DSU (disjoint set union)
    # But since we're backtracking, we need to undo operations.
    # Alternative: for each color pair, maintain the set of edges and check
    # connectivity via BFS/DFS when needed. For small graphs this is fine.
    
    # Even better: maintain for each color pair a list of edges, and a 
    # "parent" array for DSU. Since we backtrack, we can use a stack of changes.
    
    # Let's use a simpler approach: for each color pair (a,b), maintain 
    # a list of edges. When we add edge (u,v) with colors (a,b), we check 
    # if u and v are already connected in the subgraph of edges with colors 
    # {a,b}. If yes, adding this edge creates a cycle -> invalid.
    
    # To check connectivity efficiently, we'll maintain a DSU per color pair
    # with rollback capability.
    
    # Actually, let's use a different approach that's easier to implement:
    # For each color pair, maintain the number of edges and use a property:
    # A graph with V vertices and E edges is a forest iff E <= V - (number of components).
    # But we don't easily track components.
    
    # Let's just do BFS/DFS for connectivity check. With n=50 and few colors,
    # this should be fast enough.
    
    def has_bicolor_cycle(coloring, u, v, cu, cv):
        """Check if adding edge (u,v) with colors cu,cv creates a bicolored cycle.
        
        We need to check if u and v are already connected in the subgraph
        induced by vertices with colors cu or cv, using only edges where both
        endpoints have colors in {cu, cv}.
        """
        if cu == cv:
            return False  # Same color, not a bicolored pair
        
        # Find the smaller and larger color for consistent key
        c1, c2 = (cu, cv) if cu < cv else (cv, cu)
        
        # BFS from u to see if we can reach v using only edges between 
        # vertices with colors c1 or c2
        visited = set()
        queue = [u]
        visited.add(u)
        
        while queue:
            curr = queue.pop(0)
            if curr == v:
                return True  # Cycle would be created
            for neighbor in adj[curr]:
                if neighbor in visited:
                    continue
                cn = coloring[neighbor]
                cc = coloring[curr]
                # Check if this edge is part of the bicolored subgraph
                if (cc == c1 and cn == c2) or (cc == c2 and cn == c1):
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return False
    
    def backtrack(vertex, coloring, num_colors):
        """Backtracking search. Returns True if a valid coloring is found."""
        if time.time() > deadline:
            return False
        
        if vertex == n:
            return True
        
        # Try each color for this vertex
        for c in range(num_colors):
            # Check if color c is valid for vertex
            valid = True
            for neighbor in adj[vertex]:
                if coloring[neighbor] == c:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check bicolored cycle constraint
            for neighbor in adj[vertex]:
                nc = coloring[neighbor]
                if nc != c:
                    if has_bicolor_cycle(coloring, vertex, neighbor, c, nc):
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Assign color and recurse
            coloring[vertex] = c
            if backtrack(vertex + 1, coloring, num_colors):
                return True
            coloring[vertex] = -1
        
        return False
    
    # Try increasing number of colors
    for num_colors in range(3, 50):
        coloring = [-1] * n
        # Assign vertex 0 to color 0 to break symmetry
        coloring[0] = 0
        
        if backtrack(1, coloring, num_colors):
            # Found valid coloring
            return [int(c) for c in coloring]
    
    # Fallback: return a trivially valid coloring (each vertex its own color)
    return [int(i) for i in range(n)]
# EVOLVE-BLOCK-END
