"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_bicolored_forest(colors, n, edges):
    """Check that for every pair of colors, the induced subgraph is a forest."""
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = colors[u], colors[v]
        if cu == cv:
            return False
        key = (min(cu, cv), max(cu, cv))
        pair_edges[key].append((u, v))
    
    for pair, elist in pair_edges.items():
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True
        for u, v in elist:
            if not union(u, v):
                return False
    return True

def _build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

class _RollbackDSU:
    """Union-Find with rollback capability for incremental cycle detection."""
    def __init__(self, n):
        self.n = n
        self.parent = list(range(n))
        self.rank = [0] * n
        self.stack = []
    
    def find(self, x):
        while self.parent[x] != x:
            x = self.parent[x]
        return x
    
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        self.stack.append((rx, ry, self.rank[ry]))
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True
    
    def rollback(self):
        if not self.stack:
            return
        rx, ry, old_rank = self.stack.pop()
        self.parent[ry] = ry
        self.rank[rx] = old_rank

def _solve_csp(n, edges, k, deadline, rng):
    """
    CSP-based solver with MRV heuristic and forward checking.
    Uses rollback DSU for efficient incremental cycle detection.
    """
    adj = _build_adjacency(n, edges)
    colors = [-1] * n
    
    # All possible color pairs
    all_pairs = [(i, j) for i in range(k) for j in range(i+1, k)]
    pair_to_idx = {p: i for i, p in enumerate(all_pairs)}
    
    # For each pair, maintain a DSU and the set of edges currently in that pair
    pair_dsus = [_RollbackDSU(n) for _ in all_pairs]
    pair_edges = [set() for _ in all_pairs]  # (u,v) tuples
    
    # Precompute edge endpoints
    edge_list = [(u, v) for u, v in edges]
    
    # Track which vertices are assigned
    assigned = [False] * n
    unassigned_count = n
    
    # For MRV: track domain size for each unassigned vertex
    # Domain is reduced by: (1) neighbor colors, (2) cycle constraints
    
    def get_domain(v):
        """Get the set of valid colors for vertex v."""
        if assigned[v]:
            return set()
        domain = set(range(k))
        # Remove colors used by neighbors
        for nb in adj[v]:
            if assigned[nb]:
                domain.discard(colors[nb])
        return domain
    
    def check_cycle_for_pair(pair_idx, u, v):
        """Check if adding edge (u,v) to pair pair_idx creates a cycle."""
        dsu = pair_dsus[pair_idx]
        return dsu.union(u, v)
    
    def assign_vertex(v, c):
        """Assign color c to vertex v, updating DSUs. Returns True if no cycle."""
        assigned[v] = True
        colors[v] = c
        unassigned_count -= 1
        
        success = True
        rollback_points = []
        
        for nb in adj[v]:
            if not assigned[nb]:
                continue
            nb_c = colors[nb]
            if nb_c == c:
                # Should not happen due to domain filtering
                success = False
                break
            
            c1, c2 = min(c, nb_c), max(c, nb_c)
            pidx = pair_to_idx[(c1, c2)]
            
            # Check if this edge already exists in the pair
            edge_key = (min(v, nb), max(v, nb))
            if edge_key in pair_edges[pidx]:
                continue
            
            # Try to add this edge to the DSU
            if not pair_dsus[pidx].union(v, nb):
                success = False
                # Rollback this union
                pair_dsus[pidx].rollback()
                break
            
            pair_edges[pidx].add(edge_key)
            rollback_points.append((pidx, v, nb))
        
        if not success:
            # Rollback all changes
            for pidx, u, v2 in reversed(rollback_points):
                pair_edges[pidx].discard((min(u, v2), max(u, v2)))
                pair_dsus[pidx].rollback()
            assigned[v] = False
            colors[v] = -1
            unassigned_count += 1
        
        return success
    
    def unassign_vertex(v):
        """Unassign vertex v, rolling back DSU changes."""
        c = colors[v]
        assigned[v] = False
        colors[v] = -1
        unassigned_count += 1
        
        for nb in adj[v]:
            if not assigned[nb]:
                continue
            nb_c = colors[nb]
            if nb_c == c:
                continue
            
            c1, c2 = min(c, nb_c), max(c, nb_c)
            pidx = pair_to_idx[(c1, c2)]
            edge_key = (min(v, nb), max(v, nb))
            
            if edge_key in pair_edges[pidx]:
                pair_edges[pidx].discard(edge_key)
                pair_dsus[pidx].rollback()
    
    def mrv_select():
        """Select unassigned vertex with minimum remaining values."""
        best_v = -1
        best_domain_size = k + 1
        best_domain = None
        
        for v in range(n):
            if assigned[v]:
                continue
            domain = get_domain(v)
            if len(domain) < best_domain_size:
                best_domain_size = len(domain)
                best_v = v
                best_domain = domain
                if best_domain_size <= 1:
                    break
        
        return best_v, best_domain
    
    def forward_check(v, c):
        """Check if assigning c to v causes any unassigned neighbor to have empty domain."""
        for nb in adj[v]:
            if assigned[nb]:
                continue
            domain = get_domain(nb)
            domain.discard(c)
            if len(domain) == 0:
                return False
        return True
    
    def backtrack(depth):
        if time.time() > deadline:
            return "timeout"
        
        if unassigned_count == 0:
            return "success"
        
        v, domain = mrv_select()
        if v == -1:
            return "success"
        
        if len(domain) == 0:
            return "fail"
        
        # Order colors by least constraining value
        color_list = list(domain)
        rng.shuffle(color_list)
        
        for c in color_list:
            if not forward_check(v, c):
                continue
            
            if assign_vertex(v, c):
                result = backtrack(depth + 1)
                if result == "success":
                    return "success"
                if result == "timeout":
                    unassign_vertex(v)
                    return "timeout"
            
            unassign_vertex(v)
        
        return "fail"
    
    result = backtrack(0)
    if result == "success":
        return colors
    return None

def _solve_3_coloring(n, edges, deadline):
    """Solve 3-coloring with bicolored forest constraint using CSP with MRV."""
    return _solve_csp(n, edges, 3, deadline, random.Random(42))

def _solve_k_coloring(n, edges, k, deadline, rng):
    """General k-coloring solver using CSP with MRV."""
    return _solve_csp(n, edges, k, deadline, rng)

def _greedy_with_restarts(n, edges, num_colors, restarts, deadline, rng):
    """
    Greedy coloring with random restarts as fallback.
    """
    adj = _build_adjacency(n, edges)
    
    def find(parent, x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(parent, x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        parent[rx] = ry
        return True
    
    for _ in range(restarts):
        if time.time() > deadline:
            break
        
        order = list(range(n))
        rng.shuffle(order)
        
        colors = [-1] * n
        success = True
        
        for idx in order:
            if time.time() > deadline:
                success = False
                break
            
            used_colors = set()
            for nb in adj[idx]:
                if colors[nb] != -1:
                    used_colors.add(colors[nb])
            
            assigned = False
            color_order = list(range(num_colors))
            rng.shuffle(color_order)
            
            for color in color_order:
                if color in used_colors:
                    continue
                
                colors[idx] = color
                valid = True
                
                for nb in adj[idx]:
                    if colors[nb] == -1:
                        continue
                    nb_color = colors[nb]
                    c1, c2 = min(color, nb_color), max(color, nb_color)
                    
                    parent = list(range(n))
                    for u, v in edges:
                        cu, cv = colors[u], colors[v]
                        if cu == -1 or cv == -1 or cu == cv:
                            continue
                        pkey = (min(cu, cv), max(cu, cv))
                        if pkey == (c1, c2):
                            if not union(parent, u, v):
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if not valid:
                        break
                
                if valid:
                    assigned = True
                    break
            
            if not assigned:
                colors[idx] = -1
                success = False
                break
        
        if success:
            return colors
    
    return None

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 80  # Safety margin before 90s kill
    
    rng = random.Random(42)
    best = None
    best_num_colors = n
    
    # Try from 3 colors upward using CSP with MRV
    for num_colors in range(3, min(n + 1, 10)):
        if best_num_colors <= num_colors:
            break
        if time.time() > deadline:
            break
        
        result = _solve_k_coloring(n, edges, num_colors, deadline, rng)
        if result is not None:
            # Verify the solution
            if _check_bicolored_forest(result, n, edges):
                best = result
                best_num_colors = num_colors
                break
    
    # If CSP didn't find a solution, try greedy with more colors
    if best is None:
        for num_colors in range(3, min(n + 1, 15)):
            if time.time() > deadline:
                break
            result = _greedy_with_restarts(n, edges, num_colors, 20, deadline, rng)
            if result is not None and _check_bicolored_forest(result, n, edges):
                best = result
                best_num_colors = num_colors
                break
    
    if best is None:
        # Fallback: all different colors (always valid)
        best = list(range(n))
    
    # Ensure Python ints and list
    return [int(c) for c in best]
# EVOLVE-BLOCK-END
