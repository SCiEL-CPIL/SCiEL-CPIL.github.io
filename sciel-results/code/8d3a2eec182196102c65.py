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

    def has_bicolored_cycle(colors, k):
        """Check if any pair of colors induces a cycle. Returns True if cycle found."""
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
                cycle_found = False
                for u in verts:
                    for v in adj[u]:
                        if v in vs and u < v:
                            if not union(u, v):
                                cycle_found = True
                                break
                    if cycle_found:
                        break
                if cycle_found:
                    return True
        return False

    def local_search(k, max_iters=10000):
        """Min-conflicts local search for k colors."""
        colors = [random.randint(0, k - 1) for _ in range(n)]
        
        def count_conflicts():
            # Proper coloring conflicts
            proper_conflicts = 0
            for u, v in edges:
                if colors[u] == colors[v]:
                    proper_conflicts += 1
            
            # Bicolored cycle conflicts
            # We approximate by checking if any pair has a cycle.
            # For efficiency, we can just check if the graph is a forest for each pair.
            # But for local search, we need a cost function.
            # A simpler cost: number of edges in the induced subgraph that would close a cycle?
            # Actually, just checking existence of cycle is boolean.
            # Let's use a soft cost: for each pair, if it has a cycle, add penalty.
            # To make it smooth, we can count the number of redundant edges (edges that form a cycle) in the induced subgraph.
            # For a forest with V vertices, E = V - C (C components). Redundant edges = E - (V - C) = E - V + C.
            # We can compute this quickly.
            
            total_penalty = proper_conflicts * 10  # Higher weight for proper coloring
            
            for i in range(k):
                for j in range(i + 1, k):
                    verts = [x for x in range(n) if colors[x] in (i, j)]
                    if len(verts) < 3:
                        continue
                    vs = set(verts)
                    # Count edges in induced subgraph
                    e_count = 0
                    for u in verts:
                        for v in adj[u]:
                            if v in vs and u < v:
                                e_count += 1
                    # Count components
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
            
            # Pick a random vertex involved in a conflict
            # Simple strategy: pick random vertex
            v = random.randint(0, n - 1)
            
            # Try all colors, pick the one with min cost
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
                # If no improvement, maybe accept worse with some probability?
                # Or just stop if stuck?
                # For now, let's just break if we don't improve for a long time?
                # Or continue randomizing.
                pass
                
        return best_colors

    # Try local search for k=3 first
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        res = local_search(3, max_iters=5000)
        if res and verify(res, 3):
            return [int(c) for c in res]

    # Try local search for k=4
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        res = local_search(4, max_iters=5000)
        if res and verify(res, 4):
            return [int(c) for c in res]

    # Fallback to backtracking for k=3, 4
    def solve_k(k):
        """Backtracking with incremental DSU per color-pair for bicolored cycle detection."""
        colors = [-1] * n
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

        deg = [len(adj[i]) for i in range(n)]
        var_order = sorted(range(n), key=lambda x: (-deg[x], x))
        pos_in_order = [0] * n
        for i, v in enumerate(var_order):
            pos_in_order[v] = i
        nbrs = [sorted(adj[i], key=lambda x: pos_in_order[x]) for i in range(n)]

        nodes = [0]

        def bt(pos):
            if nodes[0] > 8000000:
                return False
            if time.time() > deadline - 3:
                return False
            if pos == n:
                return True

            v = var_order[pos]
            for c in range(k):
                ok = True
                for nb in nbrs[v]:
                    if colors[nb] == c:
                        ok = False
                        break
                if not ok:
                    continue

                conflict = False
                for c2 in range(k):
                    if c2 == c:
                        continue
                    nbs_c2 = [nb for nb in nbrs[v] if colors[nb] == c2]
                    if len(nbs_c2) < 2:
                        continue
                    pk = (min(c, c2), max(c, c2))
                    pd = uf[pk]
                    roots = set()
                    for nb in nbs_c2:
                        r = dsu_find(pd, nb)
                        if r in roots:
                            conflict = True
                            break
                        roots.add(r)
                    if conflict:
                        break
                if conflict:
                    continue

                colors[v] = c
                undo = []
                all_ok = True
                for c2 in range(k):
                    if c2 == c:
                        continue
                    nbs_c2 = [nb for nb in nbrs[v] if colors[nb] == c2]
                    if not nbs_c2:
                        continue
                    pk = (min(c, c2), max(c, c2))
                    pd = uf[pk]
                    for nb in nbs_c2:
                        rv = dsu_find(pd, v)
                        rn = dsu_find(pd, nb)
                        if rv == rn:
                            all_ok = False
                            break
                        undo.append((pk, rv, pd.get(rv)))
                        pd[rv] = rn
                    if not all_ok:
                        break

                if all_ok:
                    nodes[0] += 1
                    if bt(pos + 1):
                        return True

                for pk, rv, old_val in reversed(undo):
                    pd = uf[pk]
                    if old_val is None:
                        del pd[rv]
                    else:
                        pd[rv] = old_val
                colors[v] = -1

            return False

        if bt(0):
            return colors
        return None

    for k in range(3, 15):
        if time.time() > deadline - 5:
            break
        result = solve_k(k)
        if result is not None:
            if verify(result, k):
                return [int(c) for c in result]

    return [int(i) for i in range(n)]
# EVOLVE-BLOCK-END
