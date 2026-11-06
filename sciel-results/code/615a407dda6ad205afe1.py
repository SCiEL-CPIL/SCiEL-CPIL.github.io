"""Rewritten: search for larger triangle-free 4-colourings of K_n via backtracking."""
import time

# Deadline guard: keep a safety margin under the 150 s kill limit.
_DEADLINE = time.time() + 135.0


def construct():
    # --- Base: Greenwood-Gleason 4-colouring of K_16 (Clebsch graph x3) ---
    # GF(16) with primitive polynomial x^4 + x + 1 (0b10011)
    def mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log = {}
    x = 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)

    # colour(i,j) for 0<=i<j<16, no monochromatic triangle
    base16 = {}
    for i in range(16):
        for j in range(i + 1, 16):
            base16[(i, j)] = log[i ^ j] % 3

    # --- Build initial K_32: two copies, cross edges colour 3 ---
    n0 = 32
    adj = [[0] * n0 for _ in range(n0)]  # adj[i][j] = colour, 0 if i==j
    for i in range(16):
        for j in range(i + 1, 16):
            c = base16[(i, j)]
            adj[i][j] = c
            adj[j][i] = c
    for i in range(16, 32):
        for j in range(i + 1, 32):
            c = base16[(i - 16, j - 16)]
            adj[i][j] = c
            adj[j][i] = c
    for i in range(16):
        for j in range(16, 32):
            adj[i][j] = 3
            adj[j][i] = 3

    # Precompute for each vertex: list of its neighbours (colour != 0),
    # and for each colour c, the bitmask of vertices connected to v by colour c.
    # Use Python ints as bitmasks for speed.
    def build_masks(adj, n):
        masks = [[0, 0, 0, 0] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                c = adj[i][j]
                masks[i][c] |= (1 << j)
                masks[j][c] |= (1 << i)
        return masks

    masks = build_masks(adj, n0)

    # --- Try to add vertex 32 (and beyond) via backtracking ---
    def try_add_vertex(adj, masks, n):
        """Try to assign colours to all edges (v_new, v_old) for v_new = n.
        Returns True on success, modifying adj and masks in place."""
        v = n
        # For each old vertex u, the allowed colours are those c where
        # u is NOT already in a monochromatic triangle with colour c.
        # I.e., colour c is forbidden for edge (v,u) if there exists w such that
        # adj[u][w]==c and adj[v][w]==c -- but v has no edges yet.
        # Actually: colour c forbidden for (v,u) if exists w<u... no.
        # The constraint: for colour c, the set of vertices connected to v by c
        # must be independent in the c-colour subgraph. So when adding edge (v,u)
        # with colour c, we need u to have no neighbour w (already connected to v
        # by colour c) such that adj[u][w]==c.
        # Since v has no edges yet, initially all colours are allowed for all u.
        # We process vertices in order, tracking which colours are used.

        # For efficiency: for each colour c, maintain the set S_c of vertices
        # connected to v by colour c. When considering (v,u) with colour c,
        # u must not be adjacent (by colour c) to any vertex in S_c.
        # i.e., masks[u][c] & S_c == 0.

        # Order: process old vertices in a fixed order (0..n-1).
        # Use backtracking with the most-constrained-first heuristic:
        # at each step, pick the unassigned u with fewest valid colours.

        assigned = [None] * n  # assigned[u] = colour for edge (v,u)
        S = [0, 0, 0, 0]  # S[c] = bitmask of vertices assigned colour c to v

        def backtrack(idx):
            if idx == n:
                return True
            if time.time() > _DEADLINE:
                return False
            u = idx
            # Find valid colours for (v, u)
            valid = []
            for c in range(4):
                if masks[u][c] & S[c] == 0:
                    valid.append(c)
            if not valid:
                return False
            # Try colours; order by least constraining (fewest new conflicts)
            # For simplicity, try in order 0,1,2,3
            for c in valid:
                assigned[u] = c
                S[c] |= (1 << u)
                if backtrack(idx + 1):
                    return True
                S[c] &= ~(1 << u)
                assigned[u] = None
            return False

        if backtrack(0):
            # Commit
            for u in range(n):
                c = assigned[u]
                adj[v][u] = c
                adj[u][v] = c
                masks[v][c] |= (1 << u)
                masks[u][c] |= (1 << v)
            # Add row/col for new vertex
            adj.append([0] * (n + 1))
            adj[v] = [0] * (n + 1)
            adj[v][v] = 0
            for u in range(n):
                adj[v][u] = adj[u][v]
            for u in range(n):
                adj[u].append(adj[u][v])
            # Extend masks
            masks.append([0, 0, 0, 0])
            for u in range(n):
                masks[u].append(0)
            # masks[v] already set above for c values
            return True
        return False

    # Attempt to grow from 32 to 33, 34, ...
    n = n0
    # We need to extend adj and masks properly when adding vertices.
    # Let's restructure: keep adj as list of lists, rebuild masks after each add.

    # Reset to clean state
    n = 32
    adj = [[0] * n for _ in range(n)]
    for i in range(16):
        for j in range(i + 1, 16):
            c = base16[(i, j)]
            adj[i][j] = c
            adj[j][i] = c
    for i in range(16, 32):
        for j in range(i + 1, 32):
            c = base16[(i - 16, j - 16)]
            adj[i][j] = c
            adj[j][i] = c
    for i in range(16):
        for j in range(16, 32):
            adj[i][j] = 3
            adj[j][i] = 3

    def build_masks_full(adj, n):
        masks = [[0, 0, 0, 0] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                c = adj[i][j]
                masks[i][c] |= (1 << j)
                masks[j][c] |= (1 << i)
        return masks

    best_n = 32
    best_adj = [row[:] for row in adj]

    # Try to add vertices one at a time
    cur_adj = [row[:] for row in adj]
    cur_n = 32

    for target_n in range(33, 51):
        if time.time() > _DEADLINE - 5:
            break
        v = cur_n
        # Extend cur_adj to size (v+1)
        for i in range(cur_n):
            cur_adj[i].append(0)
        cur_adj.append([0] * (cur_n + 1))

        masks = build_masks_full(cur_adj, cur_n)

        # Backtracking: assign colours to edges (v, u) for u in 0..cur_n-1
        assigned = [None] * cur_n
        S = [0, 0, 0, 0]

        # Precompute: for each u, which colours are valid given current S
        # We'll use a simple recursive backtracker with MRV heuristic.

        def backtrack():
            if time.time() > _DEADLINE - 2:
                return False
            # Find unassigned u with fewest valid colours (MRV)
            best_u = -1
            best_valid = None
            best_count = 5
            for u in range(cur_n):
                if assigned[u] is not None:
                    continue
                valid = []
                for c in range(4):
                    if masks[u][c] & S[c] == 0:
                        valid.append(c)
                cnt = len(valid)
                if cnt == 0:
                    return False
                if cnt < best_count:
                    best_count = cnt
                    best_u = u
                    best_valid = valid
                    if cnt == 1:
                        break
            if best_u == -1:
                return True  # All assigned
            for c in best_valid:
                assigned[best_u] = c
                S[c] |= (1 << best_u)
                if backtrack():
                    return True
                S[c] &= ~(1 << best_u)
                assigned[best_u] = None
            return False

        if backtrack():
            # Commit the new vertex
            for u in range(cur_n):
                c = assigned[u]
                cur_adj[v][u] = c
                cur_adj[u][v] = c
            cur_n += 1
            best_n = cur_n
            best_adj = [row[:] for row in cur_adj]
        else:
            # Roll back the extension
            for i in range(cur_n):
                cur_adj[i].pop()
            cur_adj.pop()
            break

    # Build final colour list in lexicographic order
    colors = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            colors.append(best_adj[i][j])

    return {'n': best_n, 'colors': colors}
