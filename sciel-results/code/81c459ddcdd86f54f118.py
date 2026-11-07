"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']

adj = [[] for _ in range(N)]
for u, v in EDGES:
    adj[u].append(v)
    adj[v].append(u)

class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.history = []
        
    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.history.append((rx, ry, self.parent[ry], self.rank[rx], self.rank[ry]))
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

    def undo(self):
        if not self.history:
            return
        rx, ry, old_parent_ry, old_rank_rx, old_rank_ry = self.history.pop()
        self.parent[ry] = old_parent_ry
        self.rank[rx] = old_rank_rx
        self.rank[ry] = old_rank_ry

def construct():
    deadline = time.time() + 80.0
    
    # Order vertices. Symmetry breaking: 0 is fixed to 0.
    # We will try k=3, 4, 5...
    
    # Vertex order: 0 is first. Then its neighbors.
    # Simple order: 0, 1, 2, ...
    # But 0 is special.
    # Let's just use a standard order but fix 0's color.
    
    def solve(k):
        # DSUs for each pair of colors
        # Pairs are (c1, c2) with c1 < c2
        dsus = {}
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                dsus[(c1, c2)] = DSU(N)
        
        colors = [None] * N
        colors[0] = 0
        
        # Pre-check for vertex 0
        # No neighbors colored yet, so no unions needed.
        
        def backtrack(pos):
            if time.time() > deadline:
                return False
            
            if pos == N:
                return True
            
            # MRV heuristic: find the uncolored vertex with fewest valid colors
            # For simplicity, we just iterate in order, but we can optimize.
            # Given the small N, simple order might be fine if symmetry breaking is strong.
            
            v = pos
            # Try colors 0..k-1
            # Symmetry breaking for v=1: must be != 0.
            # For v=2: != 1, != 0 (if adj to 0).
            
            valid_colors = []
            for c in range(k):
                # Check adjacency
                ok = True
                for u in adj[v]:
                    if colors[u] is not None and colors[u] == c:
                        ok = False
                        break
                if ok:
                    valid_colors.append(c)
            
            if not valid_colors:
                return False
            
            # Try each valid color
            for c in valid_colors:
                colors[v] = c
                
                # Perform unions
                undo_stack = []
                valid = True
                for u in adj[v]:
                    if colors[u] is None:
                        continue
                    d = colors[u]
                    if d == c:
                        # Edge (v, u) is in subgraph for (c, e) for all e != c
                        for e in range(k):
                            if e == c:
                                continue
                            pair = (min(c, e), max(c, e))
                            dsu = dsus[pair]
                            if dsu.find(v) == dsu.find(u):
                                valid = False
                                break
                            dsu.union(v, u)
                            undo_stack.append((pair, dsu))
                        if not valid:
                            break
                    else:
                        # Edge (v, u) is in subgraph for (c, d)
                        pair = (min(c, d), max(c, d))
                        dsu = dsus[pair]
                        if dsu.find(v) == dsu.find(u):
                            valid = False
                            break
                        dsu.union(v, u)
                        undo_stack.append((pair, dsu))
                
                if valid:
                    if backtrack(pos + 1):
                        return True
                
                # Undo unions
                while undo_stack:
                    pair, dsu = undo_stack.pop()
                    dsu.undo()
                
                colors[v] = None
            
            return False
        
        return backtrack(1), colors
    
    for k in range(3, 50):
        success, colors = solve(k)
        if success:
            # Final verification
            # Check adjacency
            for u, v in EDGES:
                if colors[u] == colors[v]:
                    return [int(i) for i in range(N)] # Fallback if bug
            
            # Check forest property
            used_colors = set(colors)
            for c1 in used_colors:
                for c2 in used_colors:
                    if c1 >= c2:
                        continue
                    dsu = DSU(N)
                    for u, v in EDGES:
                        if colors[u] in (c1, c2) and colors[v] in (c1, c2):
                            if not dsu.union(u, v):
                                return [int(i) for i in range(N)] # Fallback
            return [int(c) for c in colors]
    
    # Fallback
    return [int(i) for i in range(N)]

# EVOLVE-BLOCK-END
