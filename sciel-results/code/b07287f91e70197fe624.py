"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Union-Find for bicolored cycle detection
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
    
    def check_validity(coloring, max_colors):
        """Check if coloring is valid."""
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        
        # Check bicolored subgraphs are forests
        color_pairs = set()
        for u, v in edges:
            c1, c2 = coloring[u], coloring[v]
            if c1 != c2:
                if c1 < c2:
                    color_pairs.add((c1, c2))
                else:
                    color_pairs.add((c2, c1))
        
        for c1, c2 in color_pairs:
            uf = UnionFind(n)
            for u, v in edges:
                cu, cv = coloring[u], coloring[v]
                if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                    if not uf.union(u, v):
                        return False
        
        return True
    
    def greedy_coloring(order, max_colors, randomize=False, seed=0):
        """Greedy coloring in given order."""
        import random
        if randomize:
            rng = random.Random(seed)
        else:
            rng = None
        
        coloring = [-1] * n
        for v in order:
            neighbor_colors = set(coloring[u] for u in adj[v] if coloring[u] != -1)
            available = [c for c in range(max_colors) if c not in neighbor_colors]
            if not available:
                return None
            if randomize:
                coloring[v] = rng.choice(available)
            else:
                coloring[v] = available[0]
        return coloring
    
    def mrv_coloring(max_colors, seed=0):
        """MRV (Minimum Remaining Values) heuristic coloring."""
        import random
        rng = random.Random(seed)
        
        coloring = [-1] * n
        uncolored = set(range(n))
        
        while uncolored:
            # Find vertex with fewest available colors
            best_v = None
            best_available = None
            best_count = max_colors + 1
            
            for v in uncolored:
                neighbor_colors = set(coloring[u] for u in adj[v] if coloring[u] != -1)
                available = [c for c in range(max_colors) if c not in neighbor_colors]
                if len(available) < best_count:
                    best_count = len(available)
                    best_v = v
                    best_available = available
                    if best_count == 0:
                        break
            
            if best_count == 0:
                return None
            
            coloring[best_v] = rng.choice(best_available)
            uncolored.remove(best_v)
        
        return coloring
    
    best_coloring = None
    best_count = n
    
    # Try various strategies
    # Strategy 1: Natural order greedy
    for max_colors in [3, 4]:
        if time.time() > deadline:
            break
        coloring = greedy_coloring(list(range(n)), max_colors)
        if coloring is not None and check_validity(coloring, max_colors):
            used = len(set(coloring))
            if used < best_count:
                best_count = used
                best_coloring = coloring[:]
    
    # Strategy 2: Reverse order greedy
    for max_colors in [3, 4]:
        if time.time() > deadline:
            break
        coloring = greedy_coloring(list(range(n - 1, -1, -1)), max_colors)
        if coloring is not None and check_validity(coloring, max_colors):
            used = len(set(coloring))
            if used < best_count:
                best_count = used
                best_coloring = coloring[:]
    
    # Strategy 3: Random orderings with greedy
    import random
    for max_colors in [3, 4]:
        for attempt in range(200):
            if time.time() > deadline:
                break
            order = list(range(n))
            random.seed(attempt * 1000 + max_colors)
            random.shuffle(order)
            coloring = greedy_coloring(order, max_colors)
            if coloring is not None and check_validity(coloring, max_colors):
                used = len(set(coloring))
                if used < best_count:
                    best_count = used
                    best_coloring = coloring[:]
                if best_count <= max_colors:
                    break
    
    # Strategy 4: MRV heuristic
    for max_colors in [3, 4]:
        for seed in range(200):
            if time.time() > deadline:
                break
            coloring = mrv_coloring(max_colors, seed)
            if coloring is not None and check_validity(coloring, max_colors):
                used = len(set(coloring))
                if used < best_count:
                    best_count = used
                    best_coloring = coloring[:]
                if best_count <= max_colors:
                    break
    
    # Strategy 5: Backtracking with MRV and incremental cycle detection
    def backtrack_mrv(max_colors, seed=0):
        import random
        rng = random.Random(seed)
        
        coloring = [-1] * n
        uncolored = set(range(n))
        
        # For each color pair, maintain DSU state
        # We'll store parent and rank arrays for each pair
        # But backtracking requires undo, so let's use a simpler approach:
        # Check validity at each step using BFS in the bicolored subgraph
        
        def is_valid_partial(coloring, v, c):
            """Check if assigning color c to vertex v creates a bicolored cycle."""
            coloring[v] = c
            
            # For each neighbor u with color d, check if edge (u,v) creates cycle in (c,d) subgraph
            for u in adj[v]:
                d = coloring[u]
                if d != -1 and d != c:
                    # Check if u and v are already connected in the (c,d) subgraph
                    # (excluding the edge u-v itself)
                    # Do BFS from u in the (c,d) subgraph to see if we can reach v
                    visited = set()
                    queue = [u]
                    visited.add(u)
                    while queue:
                        cur = queue.pop(0)
                        if cur == v:
                            # Found path, so adding edge creates cycle
                            coloring[v] = -1
                            return False
                        for w in adj[cur]:
                            cw = coloring[w]
                            if cw != -1 and ((cw == c and cur == v) or (cw == d and cur != v) or (cw == c and cur != v) or (cw == d and cur == v)):
                                if w not in visited and w != u:
                                    visited.add(w)
                                    queue.append(w)
            
            coloring[v] = -1
            return True
        
        def dfs(uncolored):
            if not uncolored:
                return True
            
            # MRV: pick vertex with fewest available colors
            best_v = None
            best_available = None
            best_count = max_colors + 1
            
            for v in uncolored:
                neighbor_colors = set(coloring[u] for u in adj[v] if coloring[u] != -1)
                available = [c for c in range(max_colors) if c not in neighbor_colors]
                if len(available) < best_count:
                    best_count = len(available)
                    best_v = v
                    best_available = available
                    if best_count == 0:
                        break
            
            if best_count == 0:
                return False
            
            # Shuffle available colors for randomness
            rng.shuffle(best_available)
            
            for c in best_available:
                if is_valid_partial(coloring, best_v, c):
                    coloring[best_v] = c
                    new_uncolored = uncolored - {best_v}
                    if dfs(new_uncolored):
                        return True
                    coloring[best_v] = -1
            
            return False
        
        if dfs(uncolored):
            return coloring[:]
        return None
    
    if best_count > 3:
        for seed in range(100):
            if time.time() > deadline:
                break
            coloring = backtrack_mrv(3, seed)
            if coloring is not None and check_validity(coloring, 3):
                used = len(set(coloring))
                if used < best_count:
                    best_count = used
                    best_coloring = coloring[:]
                if best_count <= 3:
                    break
    
    if best_count > 4:
        for seed in range(100):
            if time.time() > deadline:
                break
            coloring = backtrack_mrv(4, seed)
            if coloring is not None and check_validity(coloring, 4):
                used = len(set(coloring))
                if used < best_count:
                    best_count = used
                    best_coloring = coloring[:]
                if best_count <= 4:
                    break
    
    # Fallback: greedy with more colors
    if best_coloring is None:
        for max_colors in [5, 6, 7, 8]:
            coloring = greedy_coloring(list(range(n)), max_colors)
            if coloring is not None and check_validity(coloring, max_colors):
                best_coloring = coloring
                best_count = len(set(coloring))
                break
    
    if best_coloring is None:
        # Worst case fallback
        best_coloring = list(range(n))
        best_count = n
    
    # Convert to proper Python ints
    result = [int(c) for c in best_coloring]
    
    # Verify one final time
    for i, c in enumerate(result):
        assert isinstance(c, int) and not isinstance(c, bool)
        assert 0 <= c <= 49
    
    # Final validity check
    for u, v in edges:
        assert result[u] != result[v], f"Adjacent vertices {u},{v} have same color {result[u]}"
    
    # Check bicolored forests
    for c1 in range(best_count):
        for c2 in range(c1 + 1, best_count):
            uf = UnionFind(n)
            for u, v in edges:
                cu, cv = result[u], result[v]
                if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                    assert uf.union(u, v), f"Bicolored cycle in colors {c1},{c2}"
    
    return result
# EVOLVE-BLOCK-END
