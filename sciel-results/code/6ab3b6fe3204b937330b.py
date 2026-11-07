"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 80.0  # safety margin under 90s

    # Build adjacency and degree
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    deg = [len(adj[i]) for i in range(n)]

    # Variable ordering: highest degree first (MRV heuristic)
    order = sorted(range(n), key=lambda i: (-deg[i], i))
    pos = [0] * n
    for idx, v in enumerate(order):
        pos[v] = idx

    # For each vertex, list of (neighbor, edge_index) sorted by neighbor's pos (desc)
    # We'll process edges when the later-ordered endpoint gets assigned.
    # For forward checking, when assigning vertex v, we look at edges (v,w) where
    # w is already assigned (pos[w] < pos[v]) to add the bicolored edge, and we
    # must check the bicolored-forest constraint.

    # Precompute, for each vertex v, the list of already-assigned neighbors (those with pos < pos[v])
    # and the list of not-yet-assigned neighbors (pos > pos[v]) for forward checking.
    assigned_neighbors = [[] for _ in range(n)]   # neighbors with pos < pos[v]
    unassigned_neighbors = [[] for _ in range(n)] # neighbors with pos > pos[v]
    for v in range(n):
        for w in adj[v]:
            if pos[w] < pos[v]:
                assigned_neighbors[v].append(w)
            elif pos[w] > pos[v]:
                unassigned_neighbors[v].append(w)

    def _solve(k):
        # color[v] = -1 unassigned, else 0..k-1
        color = [-1] * n
        # DSU per color pair: comp[(a,b)][v] = representative of v in the bicolored
        # subgraph of colors a,b. Only meaningful for vertices colored a or b.
        # We store a dict of parent pointers per pair for only the vertices involved.
        # Use a dict: pair -> {vertex: parent}
        parent = {}   # (a,b) -> dict vertex->parent
        rank = {}     # (a,b) -> dict vertex->rank

        def find(pair, x):
            d = parent[pair]
            # path compression
            root = x
            while d[root] != root:
                root = d[root]
            while d[x] != root:
                nxt = d[x]
                d[x] = root
                x = nxt
            return root

        def union(pair, x, y):
            d = parent[pair]
            rx = find(pair, x)
            ry = find(pair, y)
            if rx == ry:
                return False  # would create a cycle
            # union by rank
            r = rank[pair]
            if r[rx] < r[ry]:
                rx, ry = ry, rx
            d[ry] = rx
            if r[rx] == r[ry]:
                r[rx] += 1
            return True

        def ensure_vertex(pair, v):
            if pair not in parent:
                parent[pair] = {}
                rank[pair] = {}
            d = parent[pair]
            if v not in d:
                d[v] = v
                rank[pair][v] = 0

        def add_edge(pair, u, v):
            # add edge between u and v in bicolored subgraph; return True if forest maintained
            ensure_vertex(pair, u)
            ensure_vertex(pair, v)
            return union(pair, u, v)

        # Undo stack: list of (pair, u, v, was_merged, root_x, root_y) to undo unions
        undo_stack = []

        def do_add_edge(pair, u, v):
            ensure_vertex(pair, u)
            ensure_vertex(pair, v)
            d = parent[pair]
            rx = find(pair, u)
            ry = find(pair, v)
            if rx == ry:
                return False
            r = rank[pair]
            if r[rx] < r[ry]:
                rx, ry = ry, rx
            # record for undo: ry will be merged into rx
            undo_stack.append((pair, ry, rx, r[rx] == r[ry]))
            d[ry] = rx
            if r[rx] == r[ry]:
                r[rx] += 1
            return True

        def undo_one():
            pair, ry, rx, inc = undo_stack.pop()
            d = parent[pair]
            r = rank[pair]
            d[ry] = ry
            if inc:
                r[rx] -= 1

        def backtrack(idx):
            if time.time() > deadline:
                return None
            if idx == n:
                return True
            v = order[idx]
            # Try each color
            for c in range(k):
                # Check adjacency constraint: no neighbor already colored c
                ok = True
                for w in assigned_neighbors[v]:
                    if color[w] == c:
                        ok = False
                        break
                if not ok:
                    continue
                # Forward check: for each unassigned neighbor w, color c must not
                # be impossible for w. Actually forward checking: after assigning v=c,
                # for each unassigned neighbor w, ensure w still has at least one
                # available color. But the bicolored constraint also matters.
                # We do a lighter forward check: for each unassigned neighbor w,
                # check that not all colors are forbidden for w.
                # Forbidden for w: colors of assigned neighbors of w (incl v now).
                # We'll just assign and let recursion catch conflicts, but add a
                # quick check to prune.
                color[v] = c
                # Add bicolored edges to already-assigned neighbors
                edge_count = 0
                valid = True
                for w in assigned_neighbors[v]:
                    cw = color[w]
                    if cw == c:
                        continue  # already checked not adjacent same color
                    a, b = (c, cw) if c < cw else (cw, c)
                    if not do_add_edge((a, b), v, w):
                        valid = False
                        break
                    edge_count += 1
                if valid:
                    # Forward check: for each unassigned neighbor, ensure at least
                    # one color remains. Simple: count available colors for w.
                    fc_ok = True
                    for w in unassigned_neighbors[v]:
                        # colors used by assigned neighbors of w
                        used = set()
                        for x in adj[w]:
                            if color[x] != -1:
                                used.add(color[x])
                        if len(used) >= k:
                            fc_ok = False
                            break
                    if fc_ok:
                        if backtrack(idx + 1):
                            return True
                # undo
                for _ in range(edge_count):
                    undo_one()
                color[v] = -1
            return None

        result = backtrack(0)
        if result:
            return color
        return None

    # Try increasing k
    for k in range(3, 50):
        if time.time() > deadline:
            break
        sol = _solve(k)
        if sol is not None:
            # verify
            assert len(set(sol)) <= k
            return [int(x) for x in sol]

    # Fallback: trivial distinct coloring (always valid)
    return list(range(n))
# EVOLVE-BLOCK-END

