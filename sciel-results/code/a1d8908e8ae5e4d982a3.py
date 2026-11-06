"""Strong edge coloring via minimum clique cover of conflict graph."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random

def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict(edges, adj):
        """Build conflict graph: vertices are edge indices, edge if conflict."""
        m = len(edges)
        # For each edge, compute its "forbidden" vertex set: endpoints + neighbors of endpoints
        forbidden = []
        for u, v in edges:
            fs = {u, v}
            fs.update(adj.get(u, ()))
            fs.update(adj.get(v, ()))
            forbidden.append(fs)
        # Conflict matrix
        conflict = [set() for _ in range(m)]
        for i in range(m):
            for j in range(i + 1, m):
                if forbidden[i] & forbidden[j]:
                    conflict[i].add(j)
                    conflict[j].add(i)
        return conflict

    def greedy_color(conflict, order=None):
        """Greedy coloring with given order (list of vertex indices)."""
        m = len(conflict)
        if order is None:
            order = list(range(m))
        color = [-1] * m
        for v in order:
            used = set()
            for u in conflict[v]:
                if color[u] >= 0:
                    used.add(color[u])
            c = 0
            while c in used:
                c += 1
            color[v] = c
        return color

    def dsatur_order(conflict, color):
        """DSATUR ordering: prioritize vertices with most saturated neighbors."""
        m = len(conflict)
        uncolored = set(range(m))
        order = []
        while uncolored:
            best = None
            best_sat = -1
            best_deg = -1
            for v in uncolored:
                sat = len(set(color[u] for u in conflict[v] if color[u] >= 0))
                deg = len(conflict[v])
                if sat > best_sat or (sat == best_sat and deg > best_deg):
                    best = v
                    best_sat = sat
                    best_deg = deg
            order.append(best)
            uncolored.remove(best)
        return order

    def greedy_dsatur(conflict):
        """Iterative DSATUR greedy coloring."""
        m = len(conflict)
        color = [-1] * m
        uncolored = set(range(m))
        while uncolored:
            # Pick vertex with max saturation, then max degree
            best = None
            best_sat = -1
            best_deg = -1
            for v in uncolored:
                sat = len(set(color[u] for u in conflict[v] if color[u] >= 0))
                deg = len(conflict[v])
                if sat > best_sat or (sat == best_sat and deg > best_deg):
                    best = v
                    best_sat = sat
                    best_deg = deg
            # Assign smallest available color
            used = set()
            for u in conflict[best]:
                if color[u] >= 0:
                    used.add(color[u])
            c = 0
            while c in used:
                c += 1
            color[best] = c
            uncolored.remove(best)
        return color

    def try_reduce_color(conflict, color, target_k, time_limit=5.0):
        """Try to recolor using at most target_k colors via backtracking."""
        m = len(conflict)
        t_start = time.time()
        # Order vertices by degree descending for better pruning
        order = sorted(range(m), key=lambda v: -len(conflict[v]))
        
        # Current color assignment
        cur_color = color[:]
        # Count colors used
        used_colors = set(c for c in cur_color if c >= 0)
        if len(used_colors) <= target_k:
            return cur_color
        
        # We need to recolor to use at most target_k colors
        # Map colors to 0..target_k-1
        color_map = {}
        for i, c in enumerate(cur_color):
            if c not in color_map:
                color_map[c] = len(color_map)
        if len(color_map) > target_k:
            return None  # Can't even map
        
        # Backtracking search
        # State: color assignment, process vertices in order
        # We'll use a recursive approach with pruning
        
        sys.setrecursionlimit(10000)
        
        def backtrack(idx):
            if time.time() - t_start > time_limit:
                return False
            if idx == m:
                return True
            v = order[idx]
            # Try each color 0..target_k-1
            for c in range(target_k):
                # Check if v can have color c
                ok = True
                for u in conflict[v]:
                    if cur_color[u] == c:
                        ok = False
                        break
                if ok:
                    cur_color[v] = c
                    if backtrack(idx + 1):
                        return True
                    cur_color[v] = -1  # undo
            return False
        
        # Initialize: try to keep existing colors where possible
        # First, try to assign each vertex its current mapped color
        for v in order:
            c = color_map.get(cur_color[v], 0)
            if c >= target_k:
                c = 0
            cur_color[v] = c
        
        # Check if current assignment is valid
        valid = True
        for v in range(m):
            for u in conflict[v]:
                if u > v and cur_color[u] == cur_color[v]:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            return cur_color
        
        # Backtracking
        if backtrack(0):
            return cur_color
        return None

    def local_search_reduce(conflict, color, max_iters=50):
        """Local search to reduce number of colors."""
        m = len(conflict)
        cur_color = color[:]
        best_color = cur_color[:]
        best_k = max(cur_color) + 1
        
        for _ in range(max_iters):
            k = max(cur_color) + 1
            # Try to eliminate one color at a time
            # Find a color class that can be merged
            improved = False
            for target in range(k - 1, -1, -1):
                # Try to recolor all vertices with color target to other colors
                vertices_in = [v for v in range(m) if cur_color[v] == target]
                if not vertices_in:
                    continue
                # Try to recolor each vertex in vertices_in to a different color
                saved = []
                success = True
                for v in vertices_in:
                    # Find available color
                    used = set()
                    for u in conflict[v]:
                        if cur_color[u] >= 0:
                            used.add(cur_color[u])
                    # Exclude target color
                    used.discard(target)
                    avail = [c for c in range(k) if c not in used]
                    if not avail:
                        success = False
                        break
                    saved.append((v, cur_color[v]))
                    cur_color[v] = avail[0]
                if success:
                    improved = True
                    if max(cur_color) + 1 < best_k:
                        best_k = max(cur_color) + 1
                        best_color = cur_color[:]
                    break
                else:
                    # Restore
                    for v, old_c in saved:
                        cur_color[v] = old_c
            
            if not improved:
                break
        
        return best_color, best_k

    def solve_instance(edges, adj):
        m = len(edges)
        if m == 0:
            return []
        conflict = build_conflict(edges, adj)
        
        # Initial greedy coloring
        color = greedy_dsatur(conflict)
        best_color = color[:]
        best_k = max(color) + 1
        
        # Local search to reduce
        best_color, best_k = local_search_reduce(conflict, best_color)
        
        # Try branch-and-bound to reduce further
        # Try to reduce by 1, 2, ... colors
        remaining_time = DEADLINE - time.time()
        if remaining_time > 2.0:
            for reduction in range(1, best_k):
                if time.time() > DEADLINE - 1.0:
                    break
                target_k = best_k - reduction
                if target_k < 2:
                    break
                t_lim = min(remaining_time * 0.3, 10.0)
                result = try_reduce_color(conflict, best_color, target_k, time_limit=t_lim)
                if result is not None:
                    best_color = result
                    best_k = max(result) + 1
                    remaining_time = DEADLINE - time.time()
                    if remaining_time < 1.0:
                        break
        
        # Final local search
        if time.time() < DEADLINE - 1.0:
            best_color, best_k = local_search_reduce(conflict, best_color, max_iters=30)
        
        return best_color

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > DEADLINE - 5.0:
            # Fallback: greedy only
            edges = instance(*spec)
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            conflict = build_conflict(edges, adj)
            color = greedy_dsatur(conflict)
            out.append(color)
            continue
        
        edges = instance(*spec)
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        color = solve_instance(edges, adj)
        out.append(color)
    
    return out
# EVOLVE-BLOCK-END
