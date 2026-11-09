"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
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
        """Check if coloring is valid: adjacent vertices differ and bicolored subgraphs are forests."""
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
    
    def try_pattern(moduli, offsets=None):
        """Try a modular pattern: color[v] = (v % moduli[i]) + offset[i]"""
        if offsets is None:
            offsets = [0] * len(moduli)
        coloring = []
        for v in range(n):
            c = 0
            for i, m in enumerate(moduli):
                c += (v % m) + offsets[i]
            coloring.append(c % 10)  # Keep colors in 0..9 range
        return coloring
    
    def local_search(coloring, max_colors, iterations=1000):
        """Local search: try to reduce number of distinct colors by recoloring vertices."""
        coloring = coloring[:]
        used_colors = set(coloring)
        
        for _ in range(iterations):
            if time.time() > deadline:
                break
            
            # Try to eliminate each unused color or merge colors
            improved = False
            
            # Try recoloring each vertex to reduce color count
            for v in range(n):
                old_c = coloring[v]
                # Try all possible new colors
                for new_c in range(max_colors):
                    if new_c == old_c:
                        continue
                    # Check if new_c is already used
                    if new_c not in used_colors and len(used_colors) >= max_colors:
                        continue
                    
                    # Check adjacency constraint
                    if any(coloring[u] == new_c for u in adj[v]):
                        continue
                    
                    # Temporarily assign
                    coloring[v] = new_c
                    
                    # Check validity
                    if check_validity(coloring, max_colors):
                        # Check if this reduces color count
                        new_used = set(coloring)
                        if len(new_used) < len(used_colors):
                            used_colors = new_used
                            improved = True
                            break
                        elif len(new_used) == len(used_colors):
                            # Same count, but might help future improvements
                            # Keep it with some probability
                            if random.random() < 0.3:
                                used_colors = new_used
                                improved = True
                                break
                    coloring[v] = old_c
                
                if improved:
                    break
            
            if not improved:
                break
        
        return coloring
    
    best_coloring = None
    best_count = n + 1
    
    # Try various modular patterns
    patterns = [
        ([5, 7], None),
        ([7, 5], None),
        ([3, 7], None),
        ([7, 3], None),
        ([5, 7, 3], None),
        ([7, 5, 3], None),
        ([11, 7], None),
        ([7, 11], None),
        ([13, 7], None),
        ([7, 13], None),
        ([5, 3], None),
        ([3, 5], None),
        ([11, 5], None),
        ([5, 11], None),
        ([13, 5], None),
        ([5, 13], None),
    ]
    
    for moduli, offsets in patterns:
        if time.time() > deadline:
            break
        coloring = try_pattern(moduli, offsets)
        if check_validity(coloring, 10):
            used = len(set(coloring))
            if used < best_count:
                best_count = used
                best_coloring = coloring[:]
    
    # Try local search from best pattern found
    if best_coloring is not None and time.time() < deadline:
        improved = local_search(best_coloring, 10, iterations=500)
        used = len(set(improved))
        if used < best_count:
            best_count = used
            best_coloring = improved
    
    # Try random initial colorings with local search
    for attempt in range(200):
        if time.time() > deadline:
            break
        random.seed(attempt * 13 + 7)
        coloring = [random.randint(0, 9) for _ in range(n)]
        # Ensure no adjacent same colors
        valid = True
        for u, v in edges:
            if coloring[u] == coloring[v]:
                valid = False
                break
        if not valid:
            continue
        if check_validity(coloring, 10):
            improved = local_search(coloring, 10, iterations=200)
            used = len(set(improved))
            if used < best_count:
                best_count = used
                best_coloring = improved
    
    # Fallback: use greedy with more colors
    if best_coloring is None:
        for max_colors in [5, 6, 7, 8, 10]:
            coloring = [-1] * n
            for v in range(n):
                neighbor_colors = set(coloring[u] for u in adj[v] if coloring[u] != -1)
                available = [c for c in range(max_colors) if c not in neighbor_colors]
                if not available:
                    break
                coloring[v] = available[0]
            if -1 not in coloring and check_validity(coloring, max_colors):
                used = len(set(coloring))
                if used < best_count:
                    best_count = used
                    best_coloring = coloring[:]
                break
    
    # Final fallback: assign unique colors
    if best_coloring is None:
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
    for c1 in range(50):
        for c2 in range(c1 + 1, 50):
            uf = UnionFind(n)
            for u, v in edges:
                cu, cv = result[u], result[v]
                if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                    assert uf.union(u, v), f"Bicolored cycle in colors {c1},{c2}"
    
    return result
# EVOLVE-BLOCK-END
