"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    deadline = time.time() + 118.0
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

    def verify(mat, n):
        """Check no monochromatic triangle."""
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

    # ---------- Bitset-based single-vertex addition (fast, from current) ----------
    def add_vertex_bitset(mat, n, time_limit, rng):
        adj = [0] * (n * 4)
        for u in range(n):
            for w in range(u + 1, n):
                c = mat[u][w]
                adj[c * n + u] |= (1 << w)
                adj[c * n + w] |= (1 << u)

        deg = [0] * n
        for u in range(n):
            for c in range(4):
                deg[u] = max(deg[u], bin(adj[c * n + u]).count('1'))
        order = sorted(range(n), key=lambda u: -deg[u])

        assign = [-1] * n
        assigned_mask = [0] * 4

        def backtrack(pos):
            if time.time() > time_limit:
                return False
            if pos == n:
                return True
            u = order[pos]
            valid = []
            for c in range(4):
                if (adj[c * n + u] & assigned_mask[c]) == 0:
                    valid.append(c)
            if not valid:
                return False
            rng.shuffle(valid)
            for c in valid:
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

    # ---------- Local search: try to improve a partial solution ----------
    def local_search(mat, n, time_limit, rng):
        """
        Given a valid n-vertex coloring, try to add vertices via local search.
        Start by adding a new vertex with random colours, then fix violations.
        """
        for attempt in range(50):
            if time.time() > time_limit:
                break
            new_n = n + 1
            new_mat = [row[:] for row in mat]
            new_mat.append([-1] * new_n)
            # Random initial colours for new vertex
            for i in range(n):
                c = rng.randint(0, 3)
                new_mat[i].append(c)
                new_mat[new_n - 1][i] = c

            # Iteratively fix violations
            improved = True
            iterations = 0
            while improved and time.time() < time_limit:
                improved = False
                iterations += 1
                if iterations > 200:
                    break
                # Find a monochromatic triangle involving the new vertex
                found = False
                for i in range(n):
                    ci = new_mat[i][new_n - 1]
                    for j in range(i + 1, n):
                        if new_mat[i][j] == ci and new_mat[j][new_n - 1] == ci:
                            # Triangle (i, j, new) of colour ci
                            # Change one of the three edges
                            choice = rng.randint(0, 2)
                            if choice == 0:
                                old = new_mat[i][j]
                                for nc in range(4):
                                    if nc == ci:
                                        continue
                                    # Check if changing (i,j) to nc creates new triangles
                                    ok = True
                                    for k in range(new_n):
                                        if k == i or k == j:
                                            continue
                                        if new_mat[i][k] == nc and new_mat[j][k] == nc:
                                            ok = False
                                            break
                                    if ok:
                                        new_mat[i][j] = nc
                                        new_mat[j][i] = nc
                                        improved = True
                                        found = True
                                        break
                            elif choice == 1:
                                old = new_mat[i][new_n - 1]
                                for nc in range(4):
                                    if nc == ci:
                                        continue
                                    ok = True
                                    for k in range(new_n):
                                        if k == i or k == new_n - 1:
                                            continue
                                        if new_mat[i][k] == nc and new_mat[k][new_n - 1] == nc:
                                            ok = False
                                            break
                                    if ok:
                                        new_mat[i][new_n - 1] = nc
                                        new_mat[new_n - 1][i] = nc
                                        improved = True
                                        found = True
                                        break
                            else:
                                old = new_mat[j][new_n - 1]
                                for nc in range(4):
                                    if nc == ci:
                                        continue
                                    ok = True
                                    for k in range(new_n):
                                        if k == j or k == new_n - 1:
                                            continue
                                        if new_mat[j][k] == nc and new_mat[k][new_n - 1] == nc:
                                            ok = False
                                            break
                                    if ok:
                                        new_mat[j][new_n - 1] = nc
                                        new_mat[new_n - 1][j] = nc
                                        improved = True
                                        found = True
                                        break
                            if found:
                                break
                    if found:
                        break

            if not found and verify(new_mat, new_n):
                return new_mat, new_n

        return mat, n

    # ---------- Main loop ----------
    current_mat = [row[:] for row in mat0]
    current_n = n0

    max_target = 51
    consecutive_fails = 0

    while current_n < max_target and time.time() < deadline - 10:
        time_limit = min(deadline - 10, time.time() + 4.0)
        new_mat, new_n = add_vertex_bitset(current_mat, current_n, time_limit, rng)

        if new_n > current_n:
            current_mat = new_mat
            current_n = new_n
            consecutive_fails = 0
            if current_n > best_n:
                best_n = current_n
                best_mat = [row[:] for row in current_mat]
        else:
            # Try local search as alternative
            time_limit2 = min(deadline - 10, time.time() + 3.0)
            new_mat2, new_n2 = local_search(current_mat, current_n, time_limit2, rng)
            if new_n2 > current_n:
                current_mat = new_mat2
                current_n = new_n2
                consecutive_fails = 0
                if current_n > best_n:
                    best_n = current_n
                    best_mat = [row[:] for row in current_mat]
            else:
                consecutive_fails += 1
                if consecutive_fails >= 10:
                    rng = random.Random(rng.randint(0, 2**31))
                    consecutive_fails = 0

    # Final polish: try a few more times with remaining time
    while current_n < max_target and time.time() < deadline - 5:
        time_limit = min(deadline - 5, time.time() + 2.0)
        new_mat, new_n = add_vertex_bitset(current_mat, current_n, time_limit, rng)
        if new_n > current_n:
            current_mat = new_mat
            current_n = new_n
            if current_n > best_n:
                best_n = current_n
                best_mat = [row[:] for row in current_mat]
        else:
            break

    best_colors = matrix_to_colors(best_mat, best_n)
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
