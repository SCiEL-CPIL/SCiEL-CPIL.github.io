"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Returns a 4-colouring of K_n with no monochromatic triangle, maximising n.
    Base: Greenwood-Gleason K16 (Clebsch graph x3) doubled to K32.
    """
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    # --- Build base K16 Greenwood-Gleason colouring ---
    # GF(16) with polynomial x^4 + x + 1 (0b10011)
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)

    # Base K16: colour of edge (i,j) = log[i^j] % 3 for i!=j, i,j in 0..15
    # (i==j gives log[0] which is undefined, but we only use i<j)
    base16 = {}
    for i in range(16):
        for j in range(i + 1, 16):
            base16[(i, j)] = log[i ^ j] % 3

    # --- Build K32: two copies of K16, cross edges all colour 3 ---
    def get_color_32(i, j):
        if i // 16 == j // 16:
            return base16[(i % 16, j % 16)]
        else:
            return 3

    # Precompute adjacency for K32
    n32 = 32
    adj = [[0] * n32 for _ in range(n32)]
    for i in range(n32):
        for j in range(i + 1, n32):
            c = get_color_32(i, j)
            adj[i][j] = c
            adj[j][i] = c

    # --- Check if a new vertex can be added ---
    # A new vertex v connects to all existing vertices.
    # For each colour c, the set of existing vertices connected to v by colour c
    # must be an independent set in the subgraph of colour c.
    # i.e., no two vertices in that set can have a colour-c edge between them.

    def can_add_vertex(adj, n, color_map):
        """
        Try to find a colour assignment for a new vertex (n) to vertices 0..n-1.
        color_map[i] = colour of edge (n, i) for i in 0..n-1.
        Returns True if valid.
        """
        # Group vertices by colour
        groups = {0: [], 1: [], 2: [], 3: []}
        for i in range(n):
            groups[color_map[i]].append(i)

        for c in range(4):
            grp = groups[c]
            for a_idx in range(len(grp)):
                for b_idx in range(a_idx + 1, len(grp)):
                    a, b = grp[a_idx], grp[b_idx]
                    if adj[a][b] == c:
                        return False
        return True

    def try_extend(adj, n, max_new=1):
        """
        Try to add up to max_new vertices one at a time.
        Returns (new_n, new_adj) or None if no extension found.
        """
        current_adj = [row[:] for row in adj]
        current_n = n

        for _ in range(max_new):
            found = False
            # Try all 4^current_n assignments? Too many for n=32.
            # Instead, use a heuristic: for each vertex i, the colour to new vertex
            # is constrained. Use backtracking with pruning.
            # Simpler: random restarts.
            for attempt in range(20000):
                if time.time() > deadline:
                    return None
                # Random assignment
                color_map = [random.randint(0, 3) for _ in range(current_n)]
                if can_add_vertex(current_adj, current_n, color_map):
                    # Add the vertex
                    new_n = current_n + 1
                    new_row = color_map[:] + [0]
                    for i in range(current_n):
                        current_adj[i].append(color_map[i])
                    current_adj.append(new_row)
                    current_n = new_n
                    found = True
                    break
            if not found:
                break

        if current_n > n:
            return current_n, current_adj
        return None

    # --- Attempt to extend K32 to K33 ---
    extended = None
    if time.time() < deadline - 5.0:
        extended = try_extend(adj, 32, max_new=1)

    # --- Build final result ---
    if extended is not None:
        n_final, adj_final = extended
        colors = []
        for i in range(n_final):
            for j in range(i + 1, n_final):
                colors.append(adj_final[i][j])
        return {'n': n_final, 'colors': colors}
    else:
        # Return K32
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(adj[i][j])
        return {'n': 32, 'colors': colors}

# EVOLVE-BLOCK-END
