import time
import random
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
edges = P['edges']
adj = defaultdict(list)
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

def check_validity(colors):
    """Verify a complete coloring is valid."""
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    used_colors = set(colors)
    if len(used_colors) < 2:
        return True
    color_list = list(used_colors)
    for i in range(len(color_list)):
        for j in range(i + 1, len(color_list)):
            a, b = color_list[i], color_list[j]
            uf = UnionFind(N)
            for u, v in edges:
                cu, cv = colors[u], colors[v]
                if (cu == a and cv == b) or (cu == b and cv == a):
                    if not uf.union(u, v):
                        return False
    return True

def backtracking_solve(K, deadline):
    """Backtracking with incremental union-find for bicolored cycle detection."""
    colors = [-1] * N
    # Track union-find for each color pair
    # Key: (min_color, max_color) -> UnionFind
    pair_ufs = {}
    # Also track which edges are in which pair's subgraph
    pair_edges = defaultdict(list)
    
    def get_uf(a, b):
        key = (min(a, b), max(a, b))
        if key not in pair_ufs:
            pair_ufs[key] = UnionFind(N)
        return pair_ufs[key]
    
    def add_edge_to_pair(u, v, a, b):
        """Add edge (u,v) to pair (a,b)'s union-find. Returns False if cycle created."""
        key = (min(a, b), max(a, b))
        if key not in pair_ufs:
            pair_ufs[key] = UnionFind(N)
        uf = pair_ufs[key]
        if not uf.union(u, v):
            return False
        pair_edges[key].append((u, v))
        return True
    
    def remove_edge_from_pair(u, v, a, b):
        """Remove edge (u,v) from pair (a,b). We rebuild the UF for simplicity."""
        key = (min(a, b), max(a, b))
        if key in pair_edges:
            pair_edges[key].remove((u, v))
            # Rebuild UF
            pair_ufs[key] = UnionFind(N)
            for eu, ev in pair_edges[key]:
                pair_ufs[key].union(eu, ev)
    
    def is_valid(v, c):
        """Check if assigning color c to vertex v is valid."""
        # Check properness
        for u in adj[v]:
            if colors[u] == c:
                return False
        
        # Check bicolored cycle constraint
        for u in adj[v]:
            cu = colors[u]
            if cu == -1 or cu == c:
                continue
            # Check if adding edge (v,u) to pair (c,cu) creates a cycle
            key = (min(c, cu), max(c, cu))
            if key in pair_ufs:
                uf = pair_ufs[key]
                if uf.find(v) == uf.find(u):
                    return False
        return True
    
    def backtrack(idx, assigned_count):
        if time.time() > deadline:
            return False
        if idx == N:
            return True
        
        v = idx  # Simple ordering for now
        
        # Try colors, prefer colors already used for smaller K
        for c in range(K):
            if is_valid(v, c):
                colors[v] = c
                # Add edges to pairs
                added_pairs = []
                valid = True
                for u in adj[v]:
                    cu = colors[u]
                    if cu != -1 and cu != c:
                        if not add_edge_to_pair(v, u, c, cu):
                            valid = False
                            break
                        added_pairs.append((u, cu))
                
                if valid:
                    if backtrack(idx + 1, assigned_count + 1):
                        return True
                
                # Undo
                for u, cu in added_pairs:
                    remove_edge_from_pair(v, u, c, cu)
                colors[v] = -1
        
        return False
    
    if backtrack(0, 0):
        return colors
    return None

def construct():
    deadline = time.time() + 55.0
    
    # Try K=3 with backtracking
    remaining = deadline - time.time()
    if remaining > 3:
        res = backtracking_solve(3, deadline - 3.0)
        if res is not None and check_validity(res):
            return [int(c) for c in res]
    
    # Try K=4 with backtracking
    remaining = deadline - time.time()
    if remaining > 5:
        res = backtracking_solve(4, deadline - 5.0)
        if res is not None and check_validity(res):
            return [int(c) for c in res]
    
    # Fallback: try a structured coloring
    # The graph is bipartite, so 2 colors work for properness
    # But bicolored forest constraint may need more
    # Try alternating pattern with 4 colors
    colors = [i % 4 for i in range(N)]
    if check_validity(colors):
        return colors
    
    # Try another structured pattern
    colors = [(i // 2) % 4 for i in range(N)]
    if check_validity(colors):
        return colors
    
    # Last resort: 5 colors
    colors = [i % 5 for i in range(N)]
    if check_validity(colors):
        return colors
    
    # Final fallback
    return list(range(N))
