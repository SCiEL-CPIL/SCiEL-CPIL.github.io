"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start = time.time()
    deadline = start + 75.0
    n = P['n']
    edges = P['edges']

    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    # vertex ordering: descending degree, then index
    order = sorted(range(n), key=lambda v: (-len(adj[v]), v))
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i

    # Precompute for each vertex: neighbors that come later in the order
    later_neighbors = [[] for _ in range(n)]
    for v in order:
        for u in adj[v]:
            if pos[u] > pos[v]:
                later_neighbors[v].append(u)

    # Backtracking state
    color = [-1] * n
    # For each color pair (c1, c2) with c1 < c2, a union-find parent dict
    # We use a dict mapping (c1, c2) -> dict of parent pointers
    uf = {}

    def find(pair_key, x):
        # pair_key is (c1, c2)
        parent = uf[pair_key]
        root = x
        while parent[root] != root:
            root = parent[root]
        # path compression
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(pair_key, a, b):
        parent = uf[pair_key]
        ra = find(pair_key, a)
        rb = find(pair_key, b)
        if ra == rb:
            return False  # would create a cycle
        parent[ra] = rb
        return True

    def make_pair_key(c1, c2):
        if c1 > c2:
            c1, c2 = c2, c1
        return (c1, c2)

    # Track best solution
    best_color = None
    best_count = n + 1

    def is_valid_solution():
        for i, v in enumerate(order):
            c = color[v]
            for u in later_neighbors[v]:
                cu = color[u]
                if c == cu:
                    return False
        return True

    def check_all_bicolored_forest():
        # Verify every color-pair subgraph is a forest using union-find from scratch
        pair_edges = {}
        for a, b in edges:
            ca, cb = color[a], color[b]
            if ca == cb:
                return False
            key = (ca, cb) if ca < cb else (cb, ca)
            pair_edges.setdefault(key, []).append((a, b))
        for key, elist in pair_edges.items():
            parent = {v: v for v in range(n)}
            # Only vertices involved
            def find2(x):
                r = x
                while parent[r] != r:
                    r = parent[r]
                while parent[x] != r:
                    parent[x], x = r, parent[x]
                return r
            for a, b in elist:
                ra, rb = find2(a), find2(b)
                if ra == rb:
                    return False
                parent[ra] = rb
        return True

    def solve(idx, used_colors, max_colors):
        nonlocal best_color, best_count
        if time.time() > deadline:
            return False

        if idx == n:
            # Found a complete solution
            if used_colors < best_count:
                best_count = used_colors
                best_color = color[:]
                return True
            return False

        v = order[idx]
        # Determine which colors are forbidden
        forbidden = set()
        for u in adj[v]:
            if color[u] != -1:
                forbidden.add(color[u])

        # Try colors 0..max_colors-1, skipping forbidden
        # Also, we want to try to keep used_colors low
        for c in range(max_colors):
            if c in forbidden:
                continue

            # Check bicolored cycle constraints with already-colored neighbors
            ok = True
            snapshots = []  # (pair_key, parent_dict_copy)
            for u in adj[v]:
                if color[u] != -1:
                    cu = color[u]
                    if cu == c:
                        ok = False
                        break
                    key = (c, cu) if c < cu else (cu, c)
                    if key in uf:
                        # Check if union would create cycle
                        parent = uf[key]
                        if find(key, v) == find(key, u):
                            ok = False
                            break
                    # If key not in uf, no prior edges, union is safe
            if not ok:
                continue

            # Apply
            color[v] = c
            # Record snapshots for rollback
            snapshots = []
            for u in adj[v]:
                if color[u] != -1:
                    cu = color[u]
                    key = (c, cu) if c < cu else (cu, c)
                    if key in uf:
                        # Save the parent dict state (shallow copy of dict)
                        snapshots.append((key, dict(uf[key])))
                    # Now perform union
                    if key not in uf:
                        uf[key] = {v: v, u: u}
                    else:
                        uf[key][v] = v
                        uf[key][u] = u
                    union(key, v, u)

            # Recurse
            new_used = used_colors
            if c >= used_colors:
                new_used = c + 1
            if solve(idx + 1, new_used, max_colors):
                return True

            # Rollback
            for key, old_parent in snapshots:
                uf[key] = old_parent
            color[v] = -1

        return False

    # Try increasing max_colors starting from 3
    for max_colors in range(3, n + 1):
        # Reset state
        color = [-1] * n
        uf = {}
        if solve(0, 0, max_colors):
            if best_color is not None:
                # Verify
                if check_all_bicolored_forest():
                    # Return the solution
                    result = [int(c) for c in best_color]
                    return result
            break
        if time.time() > deadline - 5.0:
            break

    # Fallback: return a valid solution with more colors if we found one
    if best_color is not None:
        return [int(c) for c in best_color]

    # Last resort: trivial coloring (each vertex unique) - always valid
    return list(range(n))
# EVOLVE-BLOCK-END

