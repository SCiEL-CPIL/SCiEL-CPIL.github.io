import time
import sys

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    n = P['n']
    edges = P['edges']
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Vertex ordering: static order 0..n-1 is fine, or sort by degree (all 4)
    # Let's just use 0..n-1
    
    def backtrack(k, max_colors, colors, dsus):
        if time.time() > deadline:
            return None
        
        if k == n:
            return colors[:]
            
        # Try colors 0 to max_colors-1
        for c in range(max_colors):
            # Check proper coloring
            valid = True
            for u in adj[k]:
                if colors[u] == -1:
                    continue
                if colors[u] == c:
                    valid = False
                    break
            if not valid:
                continue
                
            # Check bicolored cycle constraint
            # For each neighbor u with color d != c
            # The edge (k, u) is added to subgraph of colors {c, d}
            # We need to ensure this doesn't create a cycle in that subgraph
            
            # We need to store DSUs for each pair of colors
            # dsus is a dict: (c1, c2) -> DSU object
            # But DSU objects are mutable. We need to copy them or use a stack of changes.
            # Given n=50, copying DSUs is cheap.
            
            new_dsus = {key: val for key, val in dsus.items()}
            cycle_found = False
            
            for u in adj[k]:
                if colors[u] == -1:
                    continue
                d = colors[u]
                if d == c:
                    continue # Already checked
                
                pair = (min(c, d), max(c, d))
                if pair not in new_dsus:
                    new_dsus[pair] = DSU(n)
                
                # Union k and u in new_dsus[pair]
                # If they are already connected, adding this edge creates a cycle
                if not new_dsus[pair].union(k, u):
                    cycle_found = True
                    break
                    
            if cycle_found:
                continue
                
            colors[k] = c
            res = backtrack(k + 1, max_colors, colors, new_dsus)
            if res is not None:
                return res
            colors[k] = -1
            
        return None

    # Try 3 colors first
    for mc in [3, 4, 5, 6, 7, 8]:
        colors = [-1] * n
        dsus = {}
        res = backtrack(0, mc, colors, dsus)
        if res is not None:
            # Convert to list of ints
            return [int(x) for x in res]
            
    # Fallback: 50 colors
    return list(range(n))
