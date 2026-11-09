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

    # Main loop
    current_mat = [row[:] for row in mat0]
    current_n = n0

    max_target = 51
    consecutive_fails = 0

    while current_n < max_target and time.time() < deadline - 5:
        time_limit = min(deadline - 5, time.time() + 3.0)
        new_mat, new_n = add_vertex(current_mat, current_n, time_limit, rng)

        if new_n > current_n:
            current_mat = new_mat
            current_n = new_n
            consecutive_fails = 0
            if current_n > best_n:
                best_n = current_n
                best_mat = [row[:] for row in current_mat]
        else:
            consecutive_fails += 1
            if consecutive_fails >= 15:
                # Restart with new random seed
                rng = random.Random(rng.randint(0, 2**31))
                consecutive_fails = 0

    best_colors = matrix_to_colors(best_mat, best_n)
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
