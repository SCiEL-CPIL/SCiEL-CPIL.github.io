"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    start_time = time.time()
    deadline = start_time + 75  # 75s deadline with 15s safety margin

    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    def find(parent, x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def verify(coloring):
        """Verify a coloring is valid."""
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        colors_used = set(coloring)
        for a in colors_used:
            for b in colors_used:
                if a >= b:
                    continue
                verts = set()
                for v in range(n):
                    if coloring[v] == a or coloring[v] == b:
                        verts.add(v)
                if len(verts) > 1:
                    visited = set()
                    for start in verts:
                        if start not in visited:
                            queue = deque([start])
                            visited.add(start)
                            ec = 0
                            vc = 0
                            while queue:
                                node = queue.popleft()
                                vc += 1
                                for nb in adj[node]:
                                    if nb in verts:
                                        ec += 1
                                        if nb not in visited:
                                            visited.add(nb)
                                            queue.append(nb)
                            if ec // 2 != vc - 1:
                                return False
        return True

    def solve_with_k_colors(k, vertex_order=None):
        """Try to find a valid coloring using exactly k colors (0..k-1) using DSU-based backtracking with MRV heuristic."""
        color = [-1] * n
        solution = None

        # DSU for each color pair
        dsu = {}
        for a in range(k):
            for b in range(a + 1, k):
                dsu[(a, b)] = list(range(n))

        union_log = []

        def do_union(pair_key, u, v):
            parent = dsu[pair_key]
            ra = find(parent, u)
            rb = find(parent, v)
            if ra == rb:
                return False
            parent[ra] = rb
            union_log.append((pair_key, ra, rb))
            return True

        def undo_unions(count):
            for _ in range(count):
                pair_key, ra, rb = union_log.pop()
                dsu[pair_key][ra] = ra

        def get_mrv_unassigned():
            """Find unassigned vertex with minimum remaining values (most constrained)."""
            best_v = -1
            best_count = k + 1
            for v in range(n):
                if color[v] >= 0:
                    continue
                # Count available colors
                used_by_neighbors = set()
                for u in adj[v]:
                    if color[u] >= 0:
                        used_by_neighbors.add(color[u])
                avail = k - len(used_by_neighbors)
                if avail < best_count:
                    best_count = avail
                    best_v = v
                    if avail <= 1:
                        break
            return best_v, best_count

        def bt():
            nonlocal solution
            if time.time() > deadline:
                return False

            # Check if all assigned
            unassigned = [v for v in range(n) if color[v] < 0]
            if not unassigned:
                solution = color[:]
                return True

            # MRV: pick most constrained unassigned vertex
            v, avail = get_mrv_unassigned()
            if v < 0 or avail <= 0:
                return False

            for c in range(k):
                # Check: no neighbor has color c
                valid = True
                for u in adj[v]:
                    if color[u] == c:
                        valid = False
                        break
                if not valid:
                    continue

                color[v] = c

                conflict = False
                unions_done = 0

                for u in adj[v]:
                    cu = color[u]
                    if cu >= 0 and cu != c:
                        a, b = (c, cu) if c < cu else (cu, c)
                        pair_key = (a, b)
                        if do_union(pair_key, v, u):
                            unions_done += 1
                        else:
                            conflict = True
                            break

                if conflict:
                    undo_unions(unions_done)
                    color[v] = -1
                    continue

                if bt():
                    return True

                undo_unions(unions_done)
                color[v] = -1

            return False

        if bt():
            return solution
        return None

    # Try increasing k from 3
    for k in range(3, 6):
        if time.time() > deadline:
            break
        result = solve_with_k_colors(k)
        if result is not None:
            if verify(result):
                return [int(x) for x in result]

    # Fallback: greedy with more colors
    best_result = None
    for k in range(4, 15):
        if time.time() > deadline:
            break
        color = [-1] * n
        for v in range(n):
            used = set()
            for u in adj[v]:
                if color[u] >= 0:
                    used.add(color[u])
            for c in range(k):
                if c not in used:
                    color[v] = c
                    break
            else:
                color[v] = k - 1

        if verify(color):
            if best_result is None or len(set(color)) < len(set(best_result)):
                best_result = color[:]
            break

    if best_result is not None:
        return [int(x) for x in best_result]

    # Last resort: trivial solution
    return [int(i) for i in range(n)]

# EVOLVE-BLOCK-END
