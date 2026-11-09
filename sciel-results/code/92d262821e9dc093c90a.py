"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

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

    # ---------- Bitset-based single-vertex addition with improved heuristics ----------
    def add_vertex_bitset(mat, n, time_limit, rng):
        adj = [0] * (n * 4)
        for u in range(n):
            for w in range(u + 1, n):
                c = mat[u][w]
                adj[c * n + u] |= (1 << w)
                adj[c * n + w] |= (1 << u)

        # Compute degree per colour for better variable ordering
        deg = [[0] * n for _ in range(4)]
        for u in range(n):
            for c in range(4):
                deg[c][u] = bin(adj[c * n + u]).count('1')

        # Variable order: highest total degree first (most constrained)
        total_deg = [sum(deg[c][u] for c in range(4)) for u in range(n)]
        order = sorted(range(n), key=lambda u: -total_deg[u])

        assign = [-1] * n
        assigned_mask = [0] * 4
        nodes_visited = [0]

        def backtrack(pos):
            if time.time() > time_limit:
                return False
            nodes_visited[0] += 1
            if pos == n:
                return True
            u = order[pos]
            valid = []
            for c in range(4):
                if (adj[c * n + u] & assigned_mask[c]) == 0:
                    valid.append(c)
            if not valid:
                return False
            # Order colours by how many future vertices they conflict with (fewer = better)
            # Future vertices are those at positions > pos in order
            future_mask = 0
            for p in range(pos + 1, n):
                future_mask |= (1 << order[p])
            
            def color_score(c):
                # Count conflicts with future vertices
                conflicts = bin(adj[c * n + u] & future_mask).count('1')
                return conflicts
            
            valid.sort(key=color_score)
            # Add some randomness: with 20% probability, shuffle the last half
            if len(valid) >= 2 and rng.random() < 0.2:
                mid = len(valid) // 2
                valid[mid:] = rng.sample(valid[mid:], len(valid) - mid)
            
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

    # ---------- Improved local search: try to add a vertex via iterative repair ----------
    def local_search(mat, n, time_limit, rng):
        """
        Given a valid n-vertex coloring, try to add vertices via local search.
        Start by adding a new vertex with random colours, then fix violations.
        Uses a smarter repair strategy that prioritizes edge changes to the new vertex.
        """
        max_attempts = 100
        for attempt in range(max_attempts):
            if time.time() > time_limit:
                break
            
            new_n = n + 1
            new_mat = [row[:] for row in mat]
            new_mat.append([-1] * new_n)
            
            # Initialize new vertex edges with a heuristic: prefer colours that
            # minimize immediate conflicts
            for i in range(n):
                # Count how many vertices j have mat[i][j] == c for each c
                conflict_counts = [0] * 4
                for j in range(n):
                    if j != i:
                        conflict_counts[mat[i][j]] += 1
                # Choose colour with minimum conflicts (tie-break randomly)
                min_conf = min(conflict_counts)
                candidates = [c for c in range(4) if conflict_counts[c] == min_conf]
                c = rng.choice(candidates)
                new_mat[i].append(c)
                new_mat[new_n - 1][i] = c

            # Iteratively fix violations
            max_iters = 500
            for it in range(max_iters):
                if time.time() > time_limit:
                    break
                # Find a monochromatic triangle involving the new vertex
                found = False
                # Collect all violations first, then fix randomly chosen one
                violations = []
                for i in range(n):
                    ci = new_mat[i][new_n - 1]
                    for j in range(i + 1, n):
                        if new_mat[i][j] == ci and new_mat[j][new_n - 1] == ci:
                            violations.append((i, j, ci))
                
                if not violations:
                    break
                
                # Pick a random violation to fix
                i, j, ci = rng.choice(violations)
                
                # Try to fix by changing edge (i, new) or (j, new) first (cheaper)
                fixed = False
                # Try changing new_mat[i][new_n-1]
                for nc in range(4):
                    if nc == ci:
                        continue
                    ok = True
                    for k in range(n):
                        if k == i:
                            continue
                        if new_mat[i][k] == nc and new_mat[k][new_n - 1] == nc:
                            ok = False
                            break
                    if ok:
                        new_mat[i][new_n - 1] = nc
                        new_mat[new_n - 1][i] = nc
                        fixed = True
                        break
                
                if not fixed:
                    # Try changing new_mat[j][new_n-1]
                    for nc in range(4):
                        if nc == ci:
                            continue
                        ok = True
                        for k in range(n):
                            if k == j:
                                continue
                            if new_mat[j][k] == nc and new_mat[k][new_n - 1] == nc:
                                ok = False
                                break
                        if ok:
                            new_mat[j][new_n - 1] = nc
                            new_mat[new_n - 1][j] = nc
                            fixed = True
                            break
                
                if not fixed:
                    # Last resort: change new_mat[i][j]
                    for nc in range(4):
                        if nc == ci:
                            continue
                        ok = True
                        for k in range(n):
                            if k == i or k == j:
                                continue
                            if new_mat[i][k] == nc and new_mat[j][k] == nc:
                                ok = False
                                break
                        if ok:
                            new_mat[i][j] = nc
                            new_mat[j][i] = nc
                            fixed = True
                            break
                
                if not fixed:
                    # Stuck on this violation, try another
                    continue

            # Check if we have a valid solution
            if verify(new_mat, new_n):
                return new_mat, new_n

        return mat, n

    # ---------- Main loop with adaptive time allocation ----------
    current_mat = [row[:] for row in mat0]
    current_n = n0

    max_target = 51
    consecutive_fails = 0
    attempt_count = 0

    while current_n < max_target and time.time() < deadline - 8:
        attempt_count += 1
        # Allocate more time as n grows (harder problems)
        base_time = 3.0 + 0.1 * (current_n - n0)
        time_limit = min(deadline - 8, time.time() + base_time)
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
            time_limit2 = min(deadline - 8, time.time() + 2.5)
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
                if consecutive_fails >= 8:
                    # Reset RNG to escape local optimum
                    rng = random.Random(rng.randint(0, 2**31))
                    consecutive_fails = 0

    # Final polish: try a few more times with remaining time
    while current_n < max_target and time.time() < deadline - 3:
        time_limit = min(deadline - 3, time.time() + 1.5)
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
