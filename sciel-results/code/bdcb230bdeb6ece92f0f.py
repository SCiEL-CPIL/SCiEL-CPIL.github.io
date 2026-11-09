"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
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
    
    def check_bicolored_forests(coloring, max_colors):
        """Check that for every pair of colors used, the induced subgraph is a forest."""
        # Collect edges for each color pair
        pair_edges = defaultdict(list)
        for u, v in edges:
            cu, cv = coloring[u], coloring[v]
            if cu == cv:
                return False  # adjacent same color
            if cu > cv:
                cu, cv = cv, cu
            pair_edges[(cu, cv)].append((u, v))
        
        for (c1, c2), elist in pair_edges.items():
            uf = UnionFind(n)
            for u, v in elist:
                if not uf.union(u, v):
                    return False
        return True
    
    def count_colors(coloring):
        return len(set(coloring))
    
    best_coloring = list(range(n))
    best_count = n
    
    # Try greedy with different orderings and color counts
    for max_colors in [3, 4, 5, 6]:
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(2000):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 7 + max_colors * 13)
            order = list(range(n))
            random.shuffle(order)
            
            coloring = [-1] * n
            valid = True
            
            for v in order:
                used = set(coloring[u] for u in adj[v] if coloring[u] >= 0)
                available = [c for c in range(max_colors) if c not in used]
                if not available:
                    valid = False
                    break
                coloring[v] = random.choice(available)
            
            if not valid:
                continue
            
            if check_bicolored_forests(coloring, max_colors):
                used = count_colors(coloring)
                if used < best_count:
                    best_count = used
                    best_coloring = coloring[:]
                
                if max_colors == 3:
                    found = True
                    break
        
        if found:
            break
    
    # If best is still high, try local search to reduce color count
    if best_count > 3 and time.time() < deadline - 5:
        for attempt in range(500):
            if time.time() > deadline - 2:
                break
            
            random.seed(5000 + attempt)
            # Start from best coloring and try to reduce
            coloring = best_coloring[:]
            max_colors = best_count
            
            for step in range(100):
                if time.time() > deadline - 2:
                    break
                v = random.randint(0, n - 1)
                old_c = coloring[v]
                # Try assigning a color that's already used by a neighbor or a new lower color
                used = set(coloring[u] for u in adj[v] if coloring[u] >= 0)
                # Try colors from 0 to max_colors-1
                candidates = [c for c in range(max_colors) if c not in used]
                if not candidates:
                    continue
                new_c = random.choice(candidates)
                coloring[v] = new_c
                if check_bicolored_forests(coloring, max_colors):
                    used_count = count_colors(coloring)
                    if used_count < best_count:
                        best_count = used_count
                        best_coloring = coloring[:]
                    # Keep the change
                else:
                    coloring[v] = old_c
    
    result = [int(c) for c in best_coloring]
    
    for i, c in enumerate(result):
        assert isinstance(c, int) and not isinstance(c, bool)
        assert 0 <= c <= 49
    
    return result
# EVOLVE-BLOCK-END
