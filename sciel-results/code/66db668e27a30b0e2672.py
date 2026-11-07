"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Order vertices by degree (all degree 4 here, so use a heuristic ordering)
    # Use a simple ordering: try to order to maximize constraint propagation
    vertex_order = list(range(n))
    
    def solve_with_k(k):
        """Try to color with k colors such that no bicolored cycle exists."""
        # DSU for each color pair
        # For k colors, there are k*(k-1)/2 pairs
        # Each DSU tracks connectivity within that color pair's subgraph
        
        class DSU:
            def __init__(self, size):
                self.parent = list(range(size))
                self.rank = [0] * size
            
            def find(self, x):
                while self.parent[x] != x:
                    self.parent[x] = self.parent[self.parent[x]]
                    x = self.parent[x]
                return x
            
            def union(self, x, y):
                rx, ry = self.find(x), self.find(y)
                if rx == ry:
                    return False  # Already connected, would create cycle
                if self.rank[rx] < self.rank[ry]:
                    rx, ry = ry, rx
                self.parent[ry] = rx
                if self.rank[rx] == self.rank[ry]:
                    self.rank[rx] += 1
                return True
        
        # We need DSUs for each color pair
        # dsu[pair_index] where pair_index maps (color_a, color_b) with a < b
        pair_to_idx = {}
        idx = 0
        for a in range(k):
            for b in range(a + 1, k):
                pair_to_idx[(a, b)] = idx
                idx += 1
        num_pairs = idx
        
        # Assignment: color[v] = -1 means unassigned
        color = [-1] * n
        # For each vertex, track which colors are forbidden
        # forbidden[v] = set of colors that would create a bicolored cycle
        
        # We'll use recursive backtracking with undo
        # For efficiency, we track for each (vertex, color) whether it's valid
        
        def get_pair_idx(c1, c2):
            if c1 > c2:
                c1, c2 = c2, c1
            return pair_to_idx[(c1, c2)]
        
        # For undo: we need to save state before each assignment
        # State: for each pair, the DSU state (parent and rank arrays)
        # This is expensive, so we'll use a simpler approach:
        # Just try and if it fails, backtrack by re-solving from scratch
        # Or: use a stack of changes
        
        # Better approach: for each assignment, record what unions were done
        # and undo them
        
        def backtrack(v_idx, dsus, color_arr):
            """
            v_idx: index into vertex_order
            dsus: list of DSU objects, one per color pair
            color_arr: current color assignment
            Returns True if solution found, False otherwise
            """
            if time.time() > deadline:
                return False
            
            if v_idx == n:
                return True
            
            v = vertex_order[v_idx]
            
            # Try each color
            for c in range(k):
                # Check if color c is valid for vertex v
                valid = True
                changes = []  # (pair_idx, x, y) for each union
                
                for nb in adj[v]:
                    if color_arr[nb] == -1:
                        continue
                    nc = color_arr[nb]
                    if nc == c:
                        valid = False
                        break
                    # Check if adding edge (v, nb) with colors (c, nc) creates a cycle
                    pi = get_pair_idx(c, nc)
                    dsu = dsus[pi]
                    if not dsu.union(v, nb):
                        valid = False
                        break
                    changes.append((pi, v, nb))
                
                if not valid:
                    continue
                
                # Apply the assignment
                color_arr[v] = c
                if backtrack(v_idx + 1, dsus, color_arr):
                    return True
                
                # Undo
                color_arr[v] = -1
                for pi, x, y in reversed(changes):
                    dsu = dsus[pi]
                    # We can't easily undo DSU unions without full state saving
                    # So this approach won't work cleanly
                    # Let's use a different strategy
                
                # Actually, DSU union is not easily reversible.
                # Let me use a different approach: save full DSU state before each step
                # This is O(k^2 * n) per level which is too much.
                
                # Alternative: use a different data structure or approach
                break
            
            return False
        
        # The DSU undo problem is tricky. Let me use a different approach:
        # Instead of DSU, track for each color pair whether the subgraph is a forest
        # by checking if adding an edge creates a cycle using a simpler method.
        
        # Actually, let me use a cleaner backtracking with full state saving
        # For small k (3 or 4), the number of pairs is small (3 or 6)
        # and n=50, so saving parent arrays is feasible.
        
        def solve():
            dsus = [DSU(n) for _ in range(num_pairs)]
            color_arr = [-1] * n
            
            def bt(v_idx):
                if time.time() > deadline:
                    return None
                
                if v_idx == n:
                    return color_arr[:]
                
                v = vertex_order[v_idx]
                
                for c in range(k):
                    # Check validity
                    valid = True
                    # Record DSU states that change
                    dsu_changes = []  # (pair_idx, old_parent_copy, old_rank_copy)
                    
                    for nb in adj[v]:
                        if color_arr[nb] == -1:
                            continue
                        nc = color_arr[nb]
                        if nc == c:
                            valid = False
                            break
                        pi = get_pair_idx(c, nc)
                        dsu = dsus[pi]
                        # Check if union would create cycle
                        if dsu.find(v) == dsu.find(nb):
                            valid = False
                            break
                        # Save state before union
                        dsu_changes.append((pi, dsu.parent[:], dsu.rank[:]))
                        dsu.union(v, nb)
                    
                    if not valid:
                        # Undo any changes made
                        for pi, old_parent, old_rank in reversed(dsu_changes):
                            dsus[pi].parent = old_parent
                            dsus[pi].rank = old_rank
                        continue
                    
                    # Apply
                    color_arr[v] = c
                    result = bt(v_idx + 1)
                    if result is not None:
                        return result
                    
                    # Undo
                    color_arr[v] = -1
                    for pi, old_parent, old_rank in reversed(dsu_changes):
                        dsus[pi].parent = old_parent
                        dsus[pi].rank = old_rank
                
                return None
            
            return bt(0)
        
        return solve()
    
    # Try increasing k
    for k in range(2, 50):
        if time.time() > deadline:
            break
        result = solve_with_k(k)
        if result is not None:
            # Verify the solution
            color_arr = result
            # Check adjacent vertices differ
            for u, v in edges:
                if color_arr[u] == color_arr[v]:
                    return None  # Should not happen
            # Check no bicolored cycle
            # For each pair of colors, check the induced subgraph is a forest
            for a in range(k):
                for b in range(a + 1, k):
                    # Build subgraph on vertices with color a or b
                    # Use DSU to check for cycles
                    parent = list(range(n))
                    rank = [0] * n
                    
                    def find(x):
                        while parent[x] != x:
                            parent[x] = parent[parent[x]]
                            x = parent[x]
                        return x
                    
                    def union(x, y):
                        rx, ry = find(x), find(y)
                        if rx == ry:
                            return False
                        if rank[rx] < rank[ry]:
                            rx, ry = ry, rx
                        parent[ry] = rx
                        if rank[rx] == rank[ry]:
                            rank[rx] += 1
                        return True
                    
                    for u, v in edges:
                        if (color_arr[u] == a and color_arr[v] == b) or \
                           (color_arr[u] == b and color_arr[v] == a):
                            if not union(u, v):
                                return None  # Bicolored cycle found
            
            return [int(c) for c in color_arr]
    
    # Fallback: use all distinct colors (should always work)
    return list(range(n))

# EVOLVE-BLOCK-END
