"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 115.0
    rng = random.Random(12345)

    # ---------- GF(16) arithmetic (Greenwood-Gleason base colouring) ----------
    def mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)

    def base_color(i, j):
        if i // 16 == j // 16:
            return log[(i % 16) ^ (j % 16)] % 3
        return 3

    # ---------- initial 32-vertex colouring (two copies + bipartite colour 3) ----------
    n0 = 32
    mat0 = [[-1] * n0 for _ in range(n0)]
    for i in range(n0):
        for j in range(i + 1, n0):
            c = base_color(i, j)
            mat0[i][j] = c
            mat0[j][i] = c

    def matrix_to_colors(mat, n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(mat[i][j])
        return colors

    def is_valid_coloring(mat, n):
        """Check if the coloring has no monochromatic triangles."""
        for i in range(n):
            for j in range(i + 1, n):
                c = mat[i][j]
                if c < 0:
                    return False
                for k in range(j + 1, n):
                    if mat[i][k] == c and mat[j][k] == c:
                        return False
        return True

    best_n = n0
    best_mat = [row[:] for row in mat0]

    # ---------- Efficient vertex addition using bitsets ----------
    def add_vertex(mat, n, time_limit, rng):
        """Try to add one vertex to the n-vertex coloring using bitset-based backtracking."""
        # Build adjacency bitsets: adj[c*n + u] = bitmask of vertices w with mat[u][w]==c
        adj = [0] * (n * 4)
        for u in range(n):
            for w in range(u + 1, n):
                c = mat[u][w]
                if c >= 0:
                    adj[c * n + u] |= (1 << w)
                    adj[c * n + w] |= (1 << u)

        # Order vertices by degree (max over colors of neighbor count)
        deg = [0] * n
        for u in range(n):
            for c in range(4):
                deg[u] = max(deg[u], bin(adj[c * n + u]).count('1'))
        order = sorted(range(n), key=lambda u: -deg[u])

        assign = [-1] * n
        assigned_mask = [0] * 4  # bitmask of vertices assigned to color c

        def backtrack(pos):
            if time.time() > time_limit:
                return False
            if pos == n:
                return True

            u = order[pos]
            # Find valid colors: u must not be adjacent to any vertex in assigned_mask[c]
            valid_colors = []
            for c in range(4):
                if (adj[c * n + u] & assigned_mask[c]) == 0:
                    valid_colors.append(c)

            if not valid_colors:
                return False

            # Shuffle for randomization
            rng.shuffle(valid_colors)

            for c in valid_colors:
                assign[u] = c
                assigned_mask[c] |= (1 << u)

                if backtrack(pos + 1):
                    return True

                assigned_mask[c] &= ~(1 << u)
                assign[u] = -1

            return False

        if backtrack(0):
            new_n = n + 1
            new_mat = [row[:] for row in mat]
            new_mat.append([-1] * new_n)
            for i in range(n):
                new_mat[i].append(assign[i])
                new_mat[new_n - 1][i] = assign[i]
            return new_mat, new_n

        return mat, n

    # ---------- Local search repair for vertex addition ----------
    def local_search_add_vertex(mat, n, time_limit, rng):
        """
        Try to add a vertex using local search:
        1. Assign random initial colors to edges from new vertex
        2. Iteratively fix monochromatic triangle violations by recoloring
        """
        new_n = n + 1
        # Start with random assignment for edges from new vertex (index n)
        new_mat = [row[:] for row in mat]
        new_mat.append([-1] * new_n)
        for i in range(n):
            new_mat[i].append(rng.randint(0, 3))
            new_mat[new_n - 1][i] = new_mat[i][new_n - 1]

        # Build adjacency for existing n vertices
        adj = [0] * (n * 4)
        for u in range(n):
            for w in range(u + 1, n):
                c = mat[u][w]
                if c >= 0:
                    adj[c * n + u] |= (1 << w)
                    adj[c * n + w] |= (1 << u)

        # For the new vertex, track which colors are "forbidden" with each neighbor
        # A color c is forbidden for edge (new, u) if there exists v such that
        # mat[u][v] == c and new_mat[new][v] == c (would form triangle)
        
        def get_forbidden_colors_for_new_vertex(new_mat, n, new_idx):
            """For each vertex u, find colors that would create a triangle with new vertex."""
            forbidden = [set() for _ in range(n)]
            for u in range(n):
                for v in range(n):
                    if u == v:
                        continue
                    c_uv = mat[u][v] if u < v else mat[v][u]
                    c_new_v = new_mat[new_idx][v] if new_idx < v else new_mat[v][new_idx]
                    if c_new_v == c_uv:
                        forbidden[u].add(c_uv)
            return forbidden

        def count_violations(new_mat, n, new_idx):
            """Count number of monochromatic triangles involving the new vertex."""
            violations = 0
            for u in range(n):
                for v in range(u + 1, n):
                    c_uv = mat[u][v]
                    c_new_u = new_mat[new_idx][u]
                    c_new_v = new_mat[new_idx][v]
                    if c_new_u == c_new_v == c_uv:
                        violations += 1
            return violations

        # Local search loop
        max_iter = 5000
        for it in range(max_iter):
            if time.time() > time_limit:
                break
            
            violations = count_violations(new_mat, n, n)
            if violations == 0:
                # Success!
                return new_mat, new_n
            
            # Find a violating triangle and recolor one of the new-vertex edges
            # Find a random violating triangle
            found = False
            for _ in range(100):
                u = rng.randint(0, n - 1)
                v = rng.randint(0, n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                c_uv = mat[u][v]
                c_new_u = new_mat[n][u]
                c_new_v = new_mat[n][v]
                if c_new_u == c_new_v == c_uv:
                    # Recolor edge (new, u) to a different color
                    old_c = c_new_u
                    new_c = (old_c + 1 + rng.randint(0, 2)) % 4
                    new_mat[n][u] = new_c
                    new_mat[u][n] = new_c
                    found = True
                    break
            
            if not found:
                # Try random recoloring
                u = rng.randint(0, n - 1)
                old_c = new_mat[n][u]
                new_c = (old_c + 1 + rng.randint(0, 2)) % 4
                new_mat[n][u] = new_c
                new_mat[u][n] = new_c

        # Check if we succeeded
        if count_violations(new_mat, n, n) == 0:
            return new_mat, new_n
        
        return mat, n

    # Main loop
    current_mat = [row[:] for row in mat0]
    current_n = n0

    max_target = 51
    consecutive_fails = 0

    while current_n < max_target and time.time() < deadline - 5:
        # Try exact backtracking first with shorter time limit
        time_limit_bt = min(deadline - 5, time.time() + 2.0)
        new_mat, new_n = add_vertex(current_mat, current_n, time_limit_bt, rng)

        if new_n > current_n:
            current_mat = new_mat
            current_n = new_n
            consecutive_fails = 0
            if current_n > best_n:
                best_n = current_n
                best_mat = [row[:] for row in current_mat]
        else:
            # Try local search as fallback
            time_limit_ls = min(deadline - 5, time.time() + 3.0)
            new_mat, new_n = local_search_add_vertex(current_mat, current_n, time_limit_ls, rng)
            
            if new_n > current_n:
                # Verify the coloring is valid
                if is_valid_coloring(new_mat, new_n):
                    current_mat = new_mat
                    current_n = new_n
                    consecutive_fails = 0
                    if current_n > best_n:
                        best_n = current_n
                        best_mat = [row[:] for row in current_mat]
                else:
                    consecutive_fails += 1
            else:
                consecutive_fails += 1
            
            if consecutive_fails >= 10:
                # Restart with new random seed
                rng = random.Random(rng.randint(0, 2**31))
                consecutive_fails = 0

    best_colors = matrix_to_colors(best_mat, best_n)
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
