"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
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
    
    # Strategy 1: Try 3 colors with smart backtracking using incremental cycle check
    def backtrack_smart(max_colors, seed=0, time_limit=None):
        """Backtracking with MRV and efficient bicolored cycle detection."""
        import random
        rng = random.Random(seed)
        
        coloring = [-1] * n
        # For each color pair (c1,c2), track connected components using DSU
        # Use a dictionary mapping (c1,c2) -> list of parent arrays for rollback
        # Simpler: just do BFS check which is O(n) per edge check
        
        def creates_bicolor_cycle(v, c):
            """Check if assigning color c to vertex v creates a bicolored cycle."""
            for u in adj[v]:
                d = coloring[u]
                if d == -1 or d == c:
                    continue
                # Check if u and v are connected in the (c,d) subgraph (excluding edge u-v)
                # BFS from u in (c,d) subgraph
                visited = {u}
                queue = [u]
                while queue:
                    cur = queue.pop(0)
                    for w in adj[cur]:
                        if w == v:
                            return True  # Found path from u to v, cycle would form
                        if w in visited:
                            continue
                        cw = coloring[w]
                        if cw == -1:
                            continue
                        # w must be colored c or d, and cur must be the other
                        if (cw == c and cur != v and coloring[cur] == d) or \
                           (cw == d and cur != v and coloring[cur] == c):
                            if cur == v:
                                continue  # Skip going back to v directly
                            visited.add(w)
                            queue.append(w)
                        elif (cw == c and coloring[cur] == d) or (cw == d and coloring[cur] == c):
                            if cur != v:
                                visited.add(w)
                                queue.append(w)
            return False
        
        def dfs(depth):
            if depth == n:
                return True
            
            if time_limit is not None and time.time() > time_limit:
                return False
            
            # MRV: pick vertex with fewest available colors
            best_v = None
            best_available = None
            best_count = max_colors + 1
            
            # Pick uncolored vertex with MRV, tie-break by degree
            uncolored = [v for v in range(n) if coloring[v] == -1]
            for v in uncolored:
                neighbor_colors = set(coloring[u] for u in adj[v] if coloring[u] != -1)
                available = [c for c in range(max_colors) if c not in neighbor_colors]
                # Also filter out colors that would create bicolored cycles
                valid_available = []
                for c in available:
                    if not creates_bicolor_cycle(v, c):
                        valid_available.append(c)
                if len(valid_available) < best_count:
                    best_count = len(valid_available)
                    best_v = v
                    best_available = valid_available
                    if best_count == 0:
                        break
            
            if best_count == 0 or best_v is None:
                return False
            
            # Shuffle for randomness
            rng.shuffle(best_available)
            
            for c in best_available:
                coloring[best_v] = c
                if dfs(depth + 1):
                    return True
                coloring[best_v] = -1
            
            return False
        
        if dfs(0):
            return coloring[:]
        return None
    
    # First, try to find a 3-coloring
    for seed in range(500):
        if time.time() > deadline - 10:
            break
        tl = min(deadline - 5, start_time + 3)  # Limit each attempt to ~3s
        coloring = backtrack_smart(3, seed, tl)
        if coloring is not None:
            if check_validity(coloring, 3):
                best_coloring = coloring[:]
                best_count = len(set(coloring))
                break
    
    # If 3 colors didn't work, try 4 colors
    if best_coloring is None or best_count > 3:
        for seed in range(500):
            if time.time() > deadline - 5:
                break
            tl = min(deadline - 2, start_time + 2)  # Limit each attempt to ~2s
            coloring = backtrack_smart(4, seed, tl)
            if coloring is not None:
                if check_validity(coloring, 4):
                    used = len(set(coloring))
                    if used < best_count:
                        best_count = used
                        best_coloring = coloring[:]
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
