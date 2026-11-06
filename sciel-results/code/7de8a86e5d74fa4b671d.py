import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    def generate_instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)
    
    def build_conflict_matrix(edges, n):
        """
        Build conflict structure.
        Returns:
            - m: number of edges
            - conflict: list of lists, conflict[i] = set of j that conflict with i
            - edge_to_vertices: list of (u, v) for each edge
        """
        m = len(edges)
        edge_to_vertices = edges
        
        # Build adjacency for original graph
        orig_adj = [set() for _ in range(n)]
        for u, v in edges:
            orig_adj[u].add(v)
            orig_adj[v].add(u)
        
        # Build conflict sets
        conflict = [set() for _ in range(m)]
        for i in range(m):
            u, v = edges[i]
            for j in range(i + 1, m):
                x, y = edges[j]
                # Check conflict
                if u == x or u == y or v == x or v == y:
                    conflict[i].add(j)
                    conflict[j].add(i)
                else:
                    # Disjoint but connected by an edge
                    if (u in orig_adj[x] or u in orig_adj[y] or 
                        v in orig_adj[x] or v in orig_adj[y]):
                        conflict[i].add(j)
                        conflict[j].add(i)
                        
        return m, conflict, edge_to_vertices
    
    def greedy_first_fit(m, conflict, order=None):
        """
        Greedy first-fit coloring.
        Returns: list of colors, num_colors
        """
        if order is None:
            order = list(range(m))
        
        colors = [-1] * m
        color_masks = []
        
        for v in order:
            assigned = False
            for c, mask in enumerate(color_masks):
                valid = True
                for nb in conflict[v]:
                    if colors[nb] == c:
                        valid = False
                        break
                if valid:
                    colors[v] = c
                    color_masks[c] |= (1 << v)
                    assigned = True
                    break
            if not assigned:
                c = len(color_masks)
                colors[v] = c
                color_masks.append(1 << v)
        
        num_colors = len(color_masks)
        return colors, num_colors
    
    def backtrack_coloring(m, conflict, k, time_limit=5.0):
        """
        Backtracking with forward checking to find a coloring with at most k colors.
        Returns: list of colors if found, None otherwise.
        """
        start = time.time()
        
        # Variable ordering: most constrained first (highest degree)
        degrees = [len(conflict[v]) for v in range(m)]
        var_order = sorted(range(m), key=lambda x: -degrees[x])
        
        colors = [-1] * m
        # For each color, track which vertices have that color
        color_count = [0] * k  # count of vertices assigned to each color
        
        # For fast conflict checking: for each vertex, track which colors are used by its neighbors
        # neighbor_color_used[v][c] = True if some neighbor of v has color c
        neighbor_color_used = [[False] * k for _ in range(m)]
        
        def is_valid(v, c):
            """Check if assigning color c to vertex v is valid given current assignment."""
            return not neighbor_color_used[v][c]
        
        def assign(v, c):
            """Assign color c to vertex v and update neighbor_color_used."""
            colors[v] = c
            color_count[c] += 1
            for nb in conflict[v]:
                neighbor_color_used[nb][c] = True
        
        def unassign(v, c):
            """Unassign color c from vertex v and update neighbor_color_used."""
            colors[v] = -1
            color_count[c] -= 1
            for nb in conflict[v]:
                # Only clear if no other neighbor of nb has color c
                # Simplification: just set to False, will be recomputed if needed
                # Better: check if any other neighbor of nb has color c
                has_other = False
                for nb2 in conflict[nb]:
                    if nb2 != v and colors[nb2] == c:
                        has_other = True
                        break
                if not has_other:
                    neighbor_color_used[nb][c] = False
        
        def backtrack(idx):
            if (time.time() - start) > time_limit:
                return None
            
            if idx == m:
                return colors[:]
            
            v = var_order[idx]
            
            # Try each color
            for c in range(k):
                if is_valid(v, c):
                    # Assign
                    assign(v, c)
                    
                    result = backtrack(idx + 1)
                    if result is not None:
                        return result
                    
                    # Undo
                    unassign(v, c)
            
            return None
        
        result = backtrack(0)
        return result
    
    def solve_instance(m, conflict, time_limit=30.0):
        """
        Find minimum number of colors using binary search + backtracking.
        """
        start = time.time()
        
        # Lower bound: max degree in conflict graph + 1
        max_deg = max(len(conflict[v]) for v in range(m)) if m > 0 else 0
        lower_bound = max_deg + 1
        
        # Upper bound: greedy first fit
        degrees = [len(conflict[v]) for v in range(m)]
        order = sorted(range(m), key=lambda x: -degrees[x])
        colors_ub, ub = greedy_first_fit(m, conflict, order)
        
        # Try to improve with a few different orderings
        rng = random.Random(42)
        for _ in range(5):
            order_rand = list(range(m))
            rng.shuffle(order_rand)
            colors_try, num_try = greedy_first_fit(m, conflict, order_rand)
            if num_try < ub:
                ub = num_try
                colors_ub = colors_try
        
        # Binary search for minimum k
        best_colors = colors_ub
        best_k = ub
        
        lo, hi = lower_bound, ub
        while lo <= hi and (time.time() - start) < time_limit:
            mid = (lo + hi) // 2
            remaining = time_limit - (time.time() - start)
            if remaining < 1.0:
                break
            
            # Try to find coloring with mid colors
            bt_time = min(5.0, remaining * 0.5)
            result = backtrack_coloring(m, conflict, mid, time_limit=bt_time)
            
            if result is not None:
                best_colors = result
                best_k = mid
                hi = mid - 1
            else:
                lo = mid + 1
        
        # If binary search didn't improve, try direct search from lower_bound
        if best_k == ub and (time.time() - start) < time_limit - 2.0:
            for k in range(lower_bound, best_k):
                remaining = time_limit - (time.time() - start)
                if remaining < 2.0:
                    break
                result = backtrack_coloring(m, conflict, k, time_limit=min(5.0, remaining * 0.4))
                if result is not None:
                    best_colors = result
                    best_k = k
                    break
        
        return best_colors, best_k
    
    results = []
    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    
    # Allocate time: more time for larger instances
    total_time = deadline - time.time()
    time_budgets = [total_time * 0.25, total_time * 0.35, total_time * 0.40]
    
    for idx, (n, d, seed) in enumerate(specs):
        if time.time() > deadline - 5:
            # Fallback: quick greedy
            edges = generate_instance(n, d, seed)
            m = len(edges)
            if m == 0:
                results.append([])
                continue
            _, conflict, _ = build_conflict_matrix(edges, n)
            colors, _ = greedy_first_fit(m, conflict)
            results.append(colors)
            continue
        
        edges = generate_instance(n, d, seed)
        m = len(edges)
        
        if m == 0:
            results.append([])
            continue
        
        m, conflict, _ = build_conflict_matrix(edges, n)
        
        # Solve with backtracking + binary search
        colors, num_colors = solve_instance(m, conflict, time_limit=time_budgets[idx])
        
        results.append(colors)
    
    return results

# EVOLVE-BLOCK-END
