"""Combined program: Cayley-graph structural construction (fast guaranteed base)
plus incremental DFS backtracking with bicolored-forest pruning to minimize colors."""
import time
import math
from collections import deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _build_adjacency(edges):
    adj = [set() for _ in range(P['n'])]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _check_bicolored_forest(coloring, adj, n):
    """Exact union-find check: for every pair of colors the induced subgraph is a forest."""
    colors_used = set(coloring)
    color_list = list(colors_used)
    m = len(color_list)

    def find(x, parent):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(m):
        for j in range(i + 1, m):
            c1, c2 = color_list[i], color_list[j]
            parent = list(range(n))
            has_cycle = False
            for u in range(n):
                if coloring[u] not in (c1, c2):
                    continue
                for v in adj[u]:
                    if v <= u:
                        continue
                    if coloring[v] not in (c1, c2):
                        continue
                    ru = find(u, parent)
                    rv = find(v, parent)
                    if ru == rv:
                        has_cycle = True
                        break
                    parent[ru] = rv
                if has_cycle:
                    break
            if has_cycle:
                return False
    return True


def _cayley_coloring(k, a, n):
    """Linear Cayley coloring c(v) = (a*v) mod k. Valid iff gcd(k,a)==gcd(k,7)==1."""
    if math.gcd(k, a) != 1 or math.gcd(k, 7) != 1:
        return None
    return [int((a * v) % k) for v in range(n)]


def _backtrack_coloring(adj, n, k, deadline):
    """Incremental DFS backtracking with bicolored-forest pruning for a fixed k."""
    degrees = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: (-degrees[x], x))
    vertex_to_pos = [0] * n
    for pos, v in enumerate(order):
        vertex_to_pos[v] = pos

    # For each vertex, its earlier (already-colored) neighbors in the order
    earlier_neighbors = [[] for _ in range(n)]
    for pos in range(n):
        v = order[pos]
        for nb in adj[v]:
            if vertex_to_pos[nb] < pos:
                earlier_neighbors[pos].append(nb)

    coloring = [-1] * n

    def bt(pos):
        if time.time() > deadline:
            return None
        if pos == n:
            return list(coloring)

        v = order[pos]

        # Colors forbidden by already-colored neighbors
        neighbor_colors = set()
        for nb in earlier_neighbors[pos]:
            neighbor_colors.add(coloring[nb])

        for c in range(k):
            if c in neighbor_colors:
                continue

            coloring[v] = c

            # Incremental bicolored-forest check: adding v (color c) must not
            # create a cycle in the {c, c2} subgraph for any other color c2.
            valid = True
            for c2 in neighbor_colors:
                if c2 == c:
                    continue
                # c2-colored neighbors of v that are already colored
                c2_neighbors = [nb for nb in earlier_neighbors[pos] if coloring[nb] == c2]
                if len(c2_neighbors) >= 2:
                    # If two c2-neighbors are connected via the {c, c2} subgraph
                    # (excluding v), then adding v closes a cycle.
                    start = c2_neighbors[0]
                    targets = set(c2_neighbors)
                    targets.discard(start)
                    visited = {start}
                    stack = [start]
                    found = False
                    while stack:
                        cur = stack.pop()
                        for nb2 in adj[cur]:
                            if nb2 == v:
                                continue
                            if coloring[nb2] not in (c, c2):
                                continue
                            if nb2 in visited:
                                continue
                            if nb2 in targets:
                                found = True
                                break
                            visited.add(nb2)
                            stack.append(nb2)
                        if found:
                            break
                    if found:
                        valid = False
                        break
            if not valid:
                continue

            result = bt(pos + 1)
            if result is not None:
                return result

        coloring[v] = -1
        return None

    return bt(0)


def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill

    n = P['n']
    edges = P['edges']
    adj = _build_adjacency(edges)

    # --- Fast guaranteed base via Cayley construction (k=6, a=1) ---
    best = None
    base = _cayley_coloring(6, 1, n)
    if base is not None and _check_bicolored_forest(base, adj, n):
        best = base

    # --- Try to do better with incremental DFS backtracking ---
    # Try k from 3 upward, stopping as soon as we beat the current best.
    for k in range(3, 6):
        if best is not None and k >= len(set(best)):
            break
        if time.time() > deadline - 8.0:
            break
        coloring = _backtrack_coloring(adj, n, k, deadline - 6.0)
        if coloring is not None:
            ok = True
            for u, v in edges:
                if coloring[u] == coloring[v]:
                    ok = False
                    break
            if ok and _check_bicolored_forest(coloring, adj, n):
                best = coloring
                break  # k is increasing, first success at this k is optimal for this k

    # --- Fallback: all-distinct coloring (always valid) ---
    if best is None:
        best = [int(i) for i in range(n)]

    return [int(c) for c in best]


# EVOLVE-BLOCK-END
