# EVOLVE-BLOCK-START
import time
from collections import defaultdict
import random

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start = time.time()
    deadline = start + 85.0
    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    def verify(colors, k):
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        for i in range(k):
            for j in range(i + 1, k):
                verts = [x for x in range(n) if colors[x] in (i, j)]
                if len(verts) < 3:
                    continue
                parent = {v: v for v in verts}
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
                vs = set(verts)
                for u in verts:
                    for v in adj[u]:
                        if v in vs and u < v:
                            if not union(u, v):
                                return False
        return True

    def local_search(k, max_iters=10000):
        """Min-conflicts local search for k colors."""
        colors = [random.randint(0, k - 1) for _ in range(n)]
        
        def count_conflicts():
            proper_conflicts = 0
            for u, v in edges:
                if colors[u] == colors[v]:
                    proper_conflicts += 1
            
            total_penalty = proper_conflicts * 10
            
            for i in range(k):
                for j in range(i + 1, k):
                    verts = [x for x in range(n) if colors[x] in (i, j)]
                    if len(verts) < 3:
                        continue
                    vs = set(verts)
                    e_count = 0
                    for u in verts:
                        for v in adj[u]:
                            if v in vs and u < v:
                                e_count += 1
                    parent = {v: v for v in verts}
                    def find(x):
                        while parent[x] != x:
                            parent[x] = parent[parent[x]]
                            x = parent[x]
                        return x
                    def union(x, y):
                        rx, ry = find(x), find(y)
                        if rx != ry:
                            parent[rx] = ry
                    for u in verts:
                        for v in adj[u]:
                            if v in vs and u < v:
                                union(u, v)
                    roots = set(find(v) for v in verts)
                    c_count = len(roots)
                    v_count = len(verts)
                    redundant = e_count - v_count + c_count
                    if redundant > 0:
                        total_penalty += redundant * 5
            
            return total_penalty

        current_cost = count_conflicts()
        best_colors = colors[:]
        best_cost = current_cost
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            if current_cost == 0:
                return colors[:]
            
            v = random.randint(0, n - 1)
            
            best_c = colors[v]
            min_c = current_cost
            for c in range(k):
                if c == colors[v]:
                    continue
                old_c = colors[v]
                colors[v] = c
                new_cost = count_conflicts()
                if new_cost < min_c:
                    min_c = new_cost
                    best_c = c
                colors[v] = old_c
            
            if best_c != colors[v]:
                colors[v] = best_c
                current_cost = min_c
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
            else:
                pass
                
        return best_colors

    # SAT-based solver using DPLL
    def solve_sat(k, timeout=30):
        """Solve the bicolored-forest coloring problem using DPLL SAT."""
        # Variables:
        # x[v, c] for v in 0..n-1, c in 0..k-1: vertex v has color c
        # For bicolored cycle prevention, we use a different approach:
        # We'll use iterative deepening with constraint propagation.
        
        # Actually, let's use a smarter approach: backtracking with forward checking
        # and memoization of failed partial assignments.
        
        colors = [-1] * n
        # For each color pair (i,j), maintain a DSU to detect cycles
        uf = {}
        for i in range(k):
            for j in range(i + 1, k):
                uf[(i, j)] = {}

        def dsu_find(pd, x):
            if x not in pd:
                pd[x] = x
                return x
            root = x
            while pd[root] != root:
                root = pd[root]
            while pd[x] != root:
                nxt = pd[x]
                pd[x] = root
                x = nxt
            return root

        def dsu_union(pd, x, y):
            rx = dsu_find(pd, x)
            ry = dsu_find(pd, y)
            if rx == ry:
                return False
            pd[rx] = ry
            return True

        # Order vertices by degree (higher first) for better pruning
        deg = [len(adj[i]) for i in range(n)]
        var_order = sorted(range(n), key=lambda x: (-deg[x], x))
        
        nodes = [0]
        failed = set()  # Memoization of failed states (simplified)
        
        def can_assign(v, c):
            """Check if vertex v can be assigned color c."""
            # Check proper coloring
            for nb in adj[v]:
                if colors[nb] == c:
                    return False
            # Check bicolored cycles
            for c2 in range(k):
                if c2 == c:
                    continue
                # Find neighbors of v that have color c2
                nbs_c2 = [nb for nb in adj[v] if colors[nb] == c2]
                if len(nbs_c2) < 2:
                    continue
                pk = (min(c, c2), max(c, c2))
                pd = uf[pk]
                # Check if any two of these neighbors are already connected
                roots = set()
                for nb in nbs_c2:
                    r = dsu_find(pd, nb)
                    if r in roots:
                        return False
                    roots.add(r)
            return True

        def assign(v, c):
            """Assign color c to vertex v, updating DSUs."""
            colors[v] = c
            undo = []
            for c2 in range(k):
                if c2 == c:
                    continue
                nbs_c2 = [nb for nb in adj[v] if colors[nb] == c2]
                if not nbs_c2:
                    continue
                pk = (min(c, c2), max(c, c2))
                pd = uf[pk]
                for nb in nbs_c2:
                    rv = dsu_find(pd, v)
                    rn = dsu_find(pd, nb)
                    undo.append((pk, rv, pd.get(rv)))
                    pd[rv] = rn
            return undo

        def unassign(v, undo):
            """Undo assignment of vertex v."""
            for pk, rv, old_val in reversed(undo):
                pd = uf[pk]
                if old_val is None:
                    del pd[rv]
                else:
                    pd[rv] = old_val
            colors[v] = -1

        def bt(pos):
            if nodes[0] > 5000000:
                return False
            if time.time() > start + timeout:
                return False
            if pos == n:
                return True

            v = var_order[pos]
            
            # Try colors in order
            for c in range(k):
                if not can_assign(v, c):
                    continue
                
                undo = assign(v, c)
                nodes[0] += 1
                if bt(pos + 1):
                    return True
                unassign(v, undo)
            
            return False

        if bt(0):
            return colors[:]
        return None

    # Try local search for small k first
    for k in [3, 4, 5]:
        if time.time() > deadline - 20:
            break
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
            res = local_search(k, max_iters=3000)
            if res and verify(res, k):
                return [int(c) for c in res]

    # Try SAT/backtracking solver
    for k in range(3, 10):
        if time.time() > deadline - 5:
            break
        remaining_time = deadline - 5 - time.time()
        if remaining_time < 3:
            break
        result = solve_sat(k, timeout=min(remaining_time, 25))
        if result is not None:
            if verify(result, k):
                return [int(c) for c in result]

    # Final fallback: try one more local search with more iterations
    for k in [4, 5]:
        if time.time() > deadline - 3:
            break
        res = local_search(k, max_iters=5000)
        if res and verify(res, k):
            return [int(c) for c in res]

    # Absolute fallback
    return [int(i) for i in range(n)]
# EVOLVE-BLOCK-END
