"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def construct():
    start = time.time()
    deadline = start + 85.0
    n = P['n']
    edges = P['edges']

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    def has_bicolored_cycle(colors, k):
        """Check if any pair of colors induces a cycle."""
        for i in range(k):
            for j in range(i + 1, k):
                # Collect vertices with color i or j
                verts = [x for x in range(n) if colors[x] == i or colors[x] == j]
                if len(verts) < 3:
                    continue
                vs = set(verts)
                # Build DSU
                parent = {}
                def find(x):
                    if x not in parent:
                        parent[x] = x
                        return x
                    if parent[x] != x:
                        parent[x] = find(parent[x])
                    return parent[x]
                def union(x, y):
                    rx, ry = find(x), find(y)
                    if rx == ry:
                        return False
                    parent[rx] = ry
                    return True
                for u in verts:
                    for v in adj[u]:
                        if v in vs and u < v:
                            if not union(u, v):
                                return True
        return False

    def verify(colors, k):
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        if has_bicolored_cycle(colors, k):
            return False
        return True

    # Try small k with systematic pattern search
    # The graph looks like a 5x10 grid (rows of 7? Let me check)
    # Vertices 0-6: row 0? No, let me look at edges more carefully
    # 0 connects to 1,7,43,49
    # 1 connects to 0,2,8,44
    # 2 connects to 1,3,9,45
    # 3 connects to 2,4,10,46
    # 4 connects to 3,5,11,47
    # 5 connects to 4,6,12,48
    # 6 connects to 5,7,13,49
    # So it's a 7x? grid. 50 = 7*7 + 1? No, 50/7 ~ 7.14
    # Actually looks like a 5x10 grid with toroidal wrap
    # Let me try: vertex (r,c) = r*10 + c? 
    # 0=(0,0), neighbors: 1=(0,1), 7=(0,7)? No that doesn't work for 10-wide
    # Try 5 rows, 10 cols: v = r*10+c
    # 0=(0,0): neighbors should be (0,1)=1, (0,9)=9, (1,0)=10, (4,0)=40
    # But actual neighbors of 0: 1,7,43,49
    # 49=(4,9), 43=(4,3), 7=(0,7), 1=(0,1)
    # Doesn't match simple grid.
    
    # Let me just do a proper backtracking with constraint propagation
    # but with a much faster implementation using bitmasks and early termination

    # Precompute neighbor lists
    nbrs = [sorted(adj[i]) for i in range(n)]

    # Variable ordering: degree, then index
    deg = [len(adj[i]) for i in range(n)]
    var_order = sorted(range(n), key=lambda x: (-deg[x], x))
    pos_in_order = [0] * n
    for i, v in enumerate(var_order):
        pos_in_order[v] = i
    # Sort neighbors by position in var_order
    nbrs_ordered = [sorted(adj[i], key=lambda x: pos_in_order[x]) for i in range(n)]

    def solve_k(k, max_nodes=2000000):
        """Backtracking with incremental DSU per color pair."""
        colors = [-1] * n
        # DSU for each color pair
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
            # Path compression
            while pd[x] != root:
                next_x = pd[x]
                pd[x] = root
                x = next_x
            return root

        def dsu_union(pd, x, y):
            rx = dsu_find(pd, x)
            ry = dsu_find(pd, y)
            if rx == ry:
                return False
            pd[rx] = ry
            return True

        nodes = [0]

        def bt(pos):
            if nodes[0] > max_nodes:
                return False
            if time.time() > deadline - 5:
                return False
            if pos == n:
                return True

            v = var_order[pos]
            
            # Try colors in order
            for c in range(k):
                # Proper coloring check
                ok = True
                for nb in nbrs_ordered[v]:
                    if colors[nb] == c:
                        ok = False
                        break
                if not ok:
                    continue

                # Bicolored cycle check
                conflict = False
                undo = []
                for c2 in range(k):
                    if c2 == c:
                        continue
                    nbs_c2 = [nb for nb in nbrs_ordered[v] if colors[nb] == c2]
                    if len(nbs_c2) < 2:
                        continue
                    pk = (min(c, c2), max(c, c2))
                    pd = uf[pk]
                    # Check if any two neighbors with color c2 are in same DSU component
                    roots = set()
                    for nb in nbs_c2:
                        r = dsu_find(pd, nb)
                        if r in roots:
                            conflict = True
                            break
                        roots.add(r)
                    if conflict:
                        break
                    # Union v with each nb
                    for nb in nbs_c2:
                        rv = dsu_find(pd, v)
                        rn = dsu_find(pd, nb)
                        if rv == rn:
                            conflict = True
                            break
                        undo.append((pk, rv, pd.get(rv)))
                        pd[rv] = rn
                    if conflict:
                        break

                if conflict:
                    # Undo any partial unions
                    for pk, rv, old_val in reversed(undo):
                        pd = uf[pk]
                        if old_val is None:
                            del pd[rv]
                        else:
                            pd[rv] = old_val
                    continue

                # Assign
                colors[v] = c
                nodes[0] += 1
                if bt(pos + 1):
                    return True

                # Undo
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

    # Try k from 3 upward
    for k in range(3, 15):
        if time.time() > deadline - 10:
            break
        result = solve_k(k)
        if result is not None:
            if verify(result, k):
                return [int(c) for c in result]

    # Fallback: use more colors with a simple pattern
    # Try k=5 with modular pattern
    for k in range(4, 20):
        if time.time() > deadline - 3:
            break
        trial = [i % k for i in range(n)]
        if verify(trial, k):
            return [int(c) for c in trial]

    # Last resort: proper coloring with many colors
    return [int(i) for i in range(n)]
# EVOLVE-BLOCK-END
