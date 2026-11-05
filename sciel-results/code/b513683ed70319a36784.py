"""Combined program: Greenwood-Gleason K16 doubled base + efficient bitset-validated extension."""
import time
import random

DEADLINE = 135.0  # safety margin before 150s kill


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    # ---- Build the 32-vertex base (Greenwood-Gleason K16 doubled) ----
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log_table, x = {}, 1
    for e in range(15):
        log_table[x] = e
        x = mul(x, 2)

    base_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                base_colors.append(log_table[(i % 16) ^ (j % 16)] % 3)
            else:
                base_colors.append(3)

    best_n = 32
    best_colors = base_colors[:]

    # ---- Bitset adjacency maintenance for fast triangle checks ----
    # adj[c][i] = bitmask of vertices j such that edge (i,j) has color c
    adj = [[0] * 32 for _ in range(4)]
    idx = 0
    for i in range(32):
        for j in range(i + 1, 32):
            c = base_colors[idx]
            idx += 1
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)

    def color_of(i, j, colors, n):
        """Return color of edge (i,j), i<j, in a flat lexicographic coloring of n vertices."""
        if i > j:
            i, j = j, i
        eidx = i * (2 * n - i - 1) // 2 + (j - i - 1)
        return colors[eidx]

    def vertex_valid(assign, n, adj):
        """Check whether a new vertex (index n) with edge-color assignment assign[u] (u<n)
        creates a monochromatic triangle. For each color c, the set of u with assign[u]==c
        must be an independent set in the color-c subgraph."""
        by_color = [[], [], [], []]
        for u in range(n):
            by_color[assign[u]].append(u)
        for c in range(4):
            verts = by_color[c]
            ac = adj[c]
            for a in range(len(verts)):
                u1 = verts[a]
                for b in range(a + 1, len(verts)):
                    u2 = verts[b]
                    if (ac[u1] >> u2) & 1:
                        return False
        return True

    def add_vertex(n, colors, adj, assign):
        """Commit a new vertex at index n with edge colors assign[u] (u<n).
        Returns new n, new colors, updated adj."""
        new_n = n + 1
        # resize adj lists
        for c in range(4):
            adj[c].append(0)
        for u in range(n):
            c = assign[u]
            adj[c][u] |= (1 << n)
            adj[c][n] |= (1 << u)
        new_colors = colors + assign
        return new_n, new_colors, adj

    def try_add_vertex(n, colors, adj, deadline):
        """Attempt to add one vertex at index n using greedy + backtracking.
        Returns (success, new_n, new_colors, new_adj)."""
        # Precompute, for each color c, the color-c adjacency bitset per vertex for quick checks.
        # Greedy: assign colors to vertices 0..n-1, preferring to keep each color's set independent.
        # We'll do a backtracking search with forward checking.

        # For each vertex u, the set of colors that are 'forbidden' depends on already-assigned
        # neighbors. We process vertices in order 0..n-1.
        # For a tentative assignment, when we assign color c to vertex u, then for any already-assigned
        # vertex v (v<u) with color c, we require that (u,v) is NOT color c in the existing graph.
        # So: color c is forbidden for u if there exists v<u with assign[v]==c and color_of(u,v)==c.

        order = list(range(n))
        # Heuristic: process vertices with fewer options first is ideal, but order 0..n-1 is fine
        # and keeps the 'forbidden' check incremental.
        assign = [-1] * n
        forbidden = [[False] * 4 for _ in range(n)]

        # Precompute existing edge colors into a 2D-ish structure for O(1) lookup.
        # color_of is O(1) via formula; but to speed up, build edge_color[u][v] for u<v.
        edge_color = [[0] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_color[i][j] = edge_color[j][i] = colors[idx]
                idx += 1

        # Precompute, for each u and color c, the set of vertices v (any) where edge (u,v) has color c.
        # That's adj[c][u] restricted to < n.
        # For forward checking: when assigning color c to u, it becomes forbidden for a future vertex w
        # if edge (w,u) has color c. So forbidden_w[c] becomes True.

        # We'll do backtracking over vertices in order 0..n-1.
        # To reduce branching, at each vertex try colors in a deterministic but varied order.
        # Use recursion with a depth cap and a global trial budget.

        sys_setrecursionlimit = 10000

        def backtrack(u):
            if u == n:
                return True
            # Try each color; skip forbidden ones.
            for c in range(4):
                if forbidden[u][c]:
                    continue
                # Check: does assigning c to u conflict with any already-assigned v<u?
                # forbidden[u][c] already encodes conflicts with assigned v<u, so if not forbidden, OK.
                assign[u] = c
                # Forward check: for w>u, if edge (w,u) has color c, then color c becomes forbidden for w.
                affected = []
                ok = True
                for w in range(u + 1, n):
                    if edge_color[w][u] == c:
                        if not forbidden[w][c]:
                            forbidden[w][c] = True
                            affected.append(w)
                        else:
                            # already forbidden; fine
                            pass
                if backtrack(w if False else u + 1):
                    return True
                # undo
                for w in affected:
                    forbidden[w][c] = False
                assign[u] = -1
            return False

        # To keep it bounded and fast, limit recursion via a node counter.
        node_count = [0]
        NODE_LIMIT = 200000

        def backtrack2(u):
            if node_count[0] > NODE_LIMIT:
                return False
            node_count[0] += 1
            if u == n:
                return True
            for c in range(4):
                if forbidden[u][c]:
                    continue
                assign[u] = c
                affected = []
                for w in range(u + 1, n):
                    if edge_color[w][u] == c and not forbidden[w][c]:
                        forbidden[w][c] = True
                        affected.append(w)
                if backtrack2(u + 1):
                    return True
                for w in affected:
                    forbidden[w][c] = False
                assign[u] = -1
            return False

        # First try a fast greedy (no backtracking) to get a quick win.
        def greedy():
            g_assign = [-1] * n
            g_forbidden = [[False] * 4 for _ in range(n)]
            for u in range(n):
                chosen = -1
                for c in range(4):
                    if not g_forbidden[u][c]:
                        chosen = c
                        break
                if chosen == -1:
                    return None
                g_assign[u] = chosen
                for w in range(u + 1, n):
                    if edge_color[w][u] == chosen and not g_forbidden[w][chosen]:
                        g_forbidden[w][chosen] = True
            return g_assign

        # Try greedy first
        g = greedy()
        if g is not None and vertex_valid(g, n, adj):
            new_n, new_colors, new_adj = add_vertex(n, colors, adj, g)
            return True, new_n, new_colors, new_adj

        # Then try backtracking
        assign = [-1] * n
        forbidden = [[False] * 4 for _ in range(n)]
        node_count = [0]
        if backtrack2(0):
            if vertex_valid(assign, n, adj):
                new_n, new_colors, new_adj = add_vertex(n, colors, adj, assign)
                return True, new_n, new_colors, new_adj
        return False, n, colors, adj

    # ---- Main extension loop ----
    cur_n = best_n
    cur_colors = best_colors[:]
    cur_adj = [row[:] for row in adj]

    # We keep trying to add vertices until we can't or hit the deadline.
    # To avoid getting stuck on one unlucky ordering, we also allow random restarts of
    # the 'order' by permuting vertices, but that complicates the flat color layout.
    # Instead, we simply attempt sequential additions; if one fails, we stop (the base is
    # symmetric enough that sequential is the natural growth).
    while time.time() < deadline - 2:
        ok, cur_n, cur_colors, cur_adj = try_add_vertex(cur_n, cur_colors, cur_adj, deadline)
        if ok:
            if cur_n > best_n:
                best_n = cur_n
                best_colors = cur_colors[:]
        else:
            break

    return {'n': best_n, 'colors': best_colors}
