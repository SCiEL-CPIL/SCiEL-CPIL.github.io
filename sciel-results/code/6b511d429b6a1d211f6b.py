"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

n = P['n']
edges = P['edges']

adj = [[] for _ in range(n)]
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

def _check_bicolored_forest(coloring, k):
    """Check if for every pair of colors (c1, c2), the induced subgraph is a forest."""
    # Build DSU for each pair
    for c1 in range(k):
        for c2 in range(c1 + 1, k):
            parent = list(range(n))
            rank = [0] * n
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                px, py = find(x), find(y)
                if px == py:
                    return False
                if rank[px] < rank[py]:
                    px, py = py, px
                parent[py] = px
                if rank[px] == rank[py]:
                    rank[px] += 1
                return True
            
            for u, v in edges:
                cu, cv = coloring[u], coloring[v]
                if (cu == c1 or cu == c2) and (cv == c1 or cv == c2):
                    if not union(u, v):
                        return False
    return True

def _validate(coloring):
    """Full validation: proper coloring + bicolored forest."""
    k = max(coloring) + 1 if coloring else 0
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    return _check_bicolored_forest(coloring, k)

def _backtrack_with_constraints(k, deadline, start_time=0):
    """
    Backtracking search for k-coloring with bicolored-forest constraint.
    Uses incremental DSU per color pair with rollback.
    """
    coloring = [-1] * n
    
    # For each pair (c1, c2), maintain a DSU.
    # We'll use a list of DSUs indexed by (c1*n + c2) for c1 < c2
    # Actually, let's use a dict: key = (c1, c2) with c1 < c2
    
    # DSU with rollback
    class RollbackDSU:
        __slots__ = ['parent', 'rank', 'log']
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
            self.log = []  # (child_root, parent_root, old_rank_parent)
        
        def find(self, x):
            # No path compression for rollback
            while self.parent[x] != x:
                x = self.parent[x]
            return x
        
        def union(self, x, y):
            px, py = self.find(x), self.find(y)
            if px == py:
                return False
            if self.rank[px] < self.rank[py]:
                px, py = py, px
            # px is the new root
            self.log.append((py, px, self.rank[px]))
            self.parent[py] = px
            if self.rank[px] == self.rank[py]:
                self.rank[px] += 1
            return True
        
        def rollback(self, count):
            for _ in range(count):
                child, parent_root, old_rank = self.log.pop()
                self.parent[child] = child
                self.rank[parent_root] = old_rank
    
    # dsus: dict from (c1, c2) to RollbackDSU
    dsus = {}
    
    def get_dsu(c1, c2):
        if c1 > c2:
            c1, c2 = c2, c1
        key = (c1, c2)
        if key not in dsus:
            dsus[key] = RollbackDSU(n)
        return dsus[key]
    
    # Undo log: list of (dsu_key, num_unions)
    undo_log = []
    
    def assign(v, c):
        """
        Try to assign color c to vertex v.
        Returns list of (key, count) for undo, or None if invalid.
        """
        ops = []
        for u in adj[v]:
            d = coloring[u]
            if d == -1 or d == c:
                continue
            c1, c2 = min(c, d), max(c, d)
            dsu = get_dsu(c1, c2)
            if not dsu.union(u, v):
                # Rollback this assign
                for key, cnt in ops:
                    dsus[key].rollback(cnt)
                return None
            ops.append((c1, c2, 1))
        
        coloring[v] = c
        # Convert ops to (key, count) format
        result_ops = []
        for c1, c2, cnt in ops:
            result_ops.append(((c1, c2), cnt))
        return result_ops
    
    def unassign(v, ops):
        """Undo the assignment of color to vertex v."""
        coloring[v] = -1
        for key, cnt in ops:
            dsus[key].rollback(cnt)
    
    # Vertex ordering: use degree (all 4) then some heuristic
    # Let's try multiple orderings
    orderings = []
    
    # Ordering 1: natural
    orderings.append(list(range(n)))
    
    # Ordering 2: reverse
    orderings.append(list(range(n - 1, -1, -1)))
    
    # Ordering 3: BFS from 0
    visited = [False] * n
    queue = [0]
    visited[0] = True
    bfs_order = []
    while queue:
        u = queue.pop(0)
        bfs_order.append(u)
        for w in adj[u]:
            if not visited[w]:
                visited[w] = True
                queue.append(w)
    orderings.append(bfs_order)
    
    # Ordering 4: BFS from 25
    visited = [False] * n
    queue = [25]
    visited[25] = True
    bfs_order = []
    while queue:
        u = queue.pop(0)
        bfs_order.append(u)
        for w in adj[u]:
            if not visited[w]:
                visited[w] = True
                queue.append(w)
    orderings.append(bfs_order)
    
    best_result = None
    
    for order in orderings:
        if time.time() - start_time > deadline:
            break
        
        # Try to find a solution with this ordering
        solution = _backtrack_with_order(k, order, assign, unassign, deadline, start_time)
        if solution is not None:
            best_result = solution
            break
    
    return best_result

def _backtrack_with_order(k, order, assign_fn, unassign_fn, deadline, start_time):
    """Backtracking with a fixed vertex order."""
    coloring = [-1] * n
    n = len(order)
    
    def backtrack(idx):
        if time.time() - start_time > deadline:
            return None
        if idx == n:
            return coloring[:]
        
        v = order[idx]
        # Try colors in order
        for c in range(k):
            ops = assign_fn(v, c)
            if ops is not None:
                result = backtrack(idx + 1)
                if result is not None:
                    return result
                unassign_fn(v, ops)
        return None
    
    return backtrack(0)

def _local_search(coloring, k, deadline, start_time):
    """
    Local search: try to reduce the number of colors by recoloring.
    Start with a valid k-coloring, try to eliminate color k-1, then k-2, etc.
    """
    current = coloring[:]
    current_k = k
    
    for target_k in range(current_k - 1, 1, -1):
        if time.time() - start_time > deadline:
            break
        
        # Try to eliminate color current_k - 1
        eliminated = False
        for attempt in range(500):
            if time.time() - start_time > deadline:
                break
            
            # Find a vertex with color current_k - 1
            vertices_with_color = [v for v in range(n) if current[v] == current_k - 1]
            if not vertices_with_color:
                eliminated = True
                break
            
            # Try to recolor one of these vertices
            v = vertices_with_color[attempt % len(vertices_with_color)]
            
            # Try all colors from 0 to target_k - 1
            for c in range(target_k):
                if c == current[v]:
                    continue
                # Check if recoloring v to c is valid
                # Proper coloring: no neighbor has color c
                valid = True
                for u in adj[v]:
                    if current[u] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check bicolored forest constraint
                # Save old color
                old_c = current[v]
                current[v] = c
                
                if _check_bicolored_forest(current, current_k):
                    # Valid! Keep this change
                    eliminated = True
                    break
                else:
                    current[v] = old_c
            
            if eliminated:
                break
        
        if eliminated:
            current_k = target_k
            # Remap colors to 0..current_k-1
            color_map = {}
            next_c = 0
            for v in range(n):
                if current[v] not in color_map:
                    color_map[current[v]] = next_c
                    next_c += 1
            current = [color_map[c] for c in current]
        else:
            break
    
    return current, current_k

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    # Strategy:
    # 1. Try k=3 with backtracking
    # 2. If fails, try k=4 with backtracking  
    # 3. If fails, use greedy to get a valid coloring, then local search to reduce
    
    best_result = None
    best_k = n  # worst case
    
    # Try k=3
    for k_try in [3, 4, 5]:
        if time.time() - start_time > deadline:
            break
        
        result = _backtrack_with_constraints(k_try, deadline, start_time)
        if result is not None:
            if _validate(result):
                best_result = result
                best_k = k_try
                break
    
    # If backtracking didn't find a solution, use greedy + local search
    if best_result is None:
        # Greedy coloring
        coloring = [-1] * n
        for v in range(n):
            used = set()
            for u in adj[v]:
                if coloring[u] != -1:
                    used.add(coloring[u])
            c = 0
            while c in used:
                c += 1
            coloring[v] = c
        
        k_greedy = max(coloring) + 1
        
        # Check if greedy is valid
        if not _validate(coloring):
            # Use more colors to ensure validity
            # With enough colors, bicolored cycles are impossible
            # If each color class has at most 2 vertices, no bicolored cycle
            # Actually, let's just use a proper coloring with enough colors
            # that the induced subgraph for any pair is a tree or forest
            
            # Safest: use n colors (each vertex unique)
            # But let's try to be smarter
            # Use a coloring where each color appears at most 2 times
            # and the two vertices of the same color are not in a cycle together
            
            # For now, fallback to n colors if needed
            if k_greedy > 8:
                # Try to reduce
                pass
        
        # Local search to reduce colors
        if _validate(coloring):
            improved, improved_k = _local_search(coloring, k_greedy, deadline, start_time)
            if _validate(improved):
                best_result = improved
                best_k = improved_k
        else:
            # Need a valid coloring. Use unique colors as fallback
            best_result = list(range(n))
            best_k = n
    
    # Final validation
    if best_result is not None and _validate(best_result):
        # Ensure types are Python int
        return [int(c) for c in best_result]
    
    # Absolute fallback
    return [int(i) for i in range(n)]

# EVOLVE-BLOCK-END
