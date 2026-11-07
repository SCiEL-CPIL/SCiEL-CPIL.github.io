"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 50  # 50s safety margin within 60s limit
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Union-Find for bicolored cycle detection
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
                return False  # cycle detected
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            return True
    
    def validate(coloring, max_color):
        """Check if coloring is valid: proper and no bicolored cycles."""
        # Proper coloring check
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        # Bicolored cycle check using union-find per color pair
        # For each pair of colors, check if edges between them form a forest
        color_pairs = {}
        for u, v in edges:
            c1, c2 = coloring[u], coloring[v]
            if c1 > c2:
                c1, c2 = c2, c1
            if c1 == c2:
                continue  # already checked above
            key = (c1, c2)
            if key not in color_pairs:
                color_pairs[key] = []
            color_pairs[key].append((u, v))
        
        for (c1, c2), edge_list in color_pairs.items():
            dsu = DSU(n)
            for u, v in edge_list:
                if not dsu.union(u, v):
                    return False  # bicolored cycle found
        return True
    
    # Backtracking with incremental validation
    def solve(max_colors, order):
        """Try to color with at most max_colors colors."""
        colors = [-1] * n
        # Track edges per color pair for incremental bicolored cycle detection
        # We'll use a simpler approach: validate after each assignment using union-find
        # But that's expensive. Instead, we track connectivity per color pair.
        
        # For efficiency, use full validation but with early termination
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            
            v = order[idx]
            # Try colors in order of least constraining value
            available = set(range(max_colors))
            for nb in adj[v]:
                if colors[nb] != -1:
                    available.discard(colors[nb])
            
            for c in sorted(available, key=lambda x: -sum(1 for nb in adj[v] if colors[nb] == x)):
                colors[v] = c
                # Quick check: does this create a bicolored cycle?
                # Only check edges from v to already-colored neighbors
                cycle_found = False
                # For each colored neighbor, check the color pair
                pairs_to_check = {}
                for nb in adj[v]:
                    if colors[nb] != -1:
                        c1, c2 = c, colors[nb]
                        if c1 > c2:
                            c1, c2 = c2, c1
                        pairs_to_check.setdefault((c1, c2), []).append((v, nb))
                
                # Check if adding these edges creates cycles
                for (c1, c2), elist in pairs_to_check.items():
                    dsu = DSU(n)
                    # Add all existing edges of this color pair
                    for u, w in edges:
                        cu, cw = colors[u], colors[w]
                        if cu != -1 and cw != -1:
                            cc1, cc2 = cu, cw
                            if cc1 > cc2:
                                cc1, cc2 = cc2, cc1
                            if cc1 == c1 and cc2 == c2:
                                dsu.union(u, w)
                    # Now add the new edges
                    for u, w in elist:
                        if not dsu.union(u, w):
                            cycle_found = True
                            break
                    if cycle_found:
                        break
                
                if not cycle_found:
                    if backtrack(idx + 1):
                        return True
                
                colors[v] = -1
            
            return False
        
        return backtrack(0)
    
    # Variable ordering: degree-based (highest degree first)
    degree = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: -degree[x])
    
    # Try increasing number of colors
    for max_colors in range(2, 10):
        result = solve(max_colors, order)
        if result:
            # Extract actual colors used
            used = set()
            for c in colors:
                if c != -1:
                    used.add(c)
            # Remap to 0..k-1
            color_map = {}
            next_color = 0
            for c in sorted(used):
                color_map[c] = next_color
                next_color += 1
            final = [color_map[c] for c in colors]
            # Final validation
            if validate(final, max_colors):
                return [int(x) for x in final]
    
    # Fallback: return a valid coloring with more colors
    # Greedy coloring with validation
    colors = [0] * n
    for v in order:
        used = set()
        for nb in adj[v]:
            if colors[nb] != -1:
                used.add(colors[nb])
        c = 0
        while c in used:
            c += 1
        colors[v] = c
    
    # If greedy doesn't work well, just return range as fallback
    return [int(i) for i in range(n)]

# EVOLVE-BLOCK-END
