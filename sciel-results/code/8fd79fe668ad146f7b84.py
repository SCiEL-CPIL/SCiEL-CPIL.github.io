import time
import random
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = t0 + 118.0

    def edge_idx(i, j, n):
        return i * n - i * (i + 1) // 2 + (j - i - 1)

    def build_seed_32():
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

        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                if i // 16 == j // 16:
                    colors.append(log[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors.append(3)
        return colors

    seed_32 = build_seed_32()

    # CSP approach: add vertices with constraint propagation
    def try_add_vertex_csp(n, colors, deadline):
        """Try to add vertex n using CSP with forward checking. Returns list of n colors or None."""
        # For the new vertex v=n, we assign colors to edges (i, v) for i=0..n-1
        # Constraint: for any i<j<n, if color(i,v)==color(j,v)==c, then color(i,j) != c
        
        assignment = [None] * n
        # domains[v] = set of valid colors for edge (v, new_vertex)
        domains = [set([0, 1, 2, 3]) for _ in range(n)]
        
        # Precompute: for each pair (i,j), what color is edge (i,j)
        edge_colors = {}
        for i in range(n):
            for j in range(i + 1, n):
                edge_colors[(i, j)] = colors[edge_idx(i, j, n)]
        
        def backtrack(v):
            if time.time() > deadline:
                return False
            if v == n:
                return True
            
            # Forward checking: try each color in domain
            for c in sorted(domains[v]):
                assignment[v] = c
                ok = True
                # Check against already assigned vertices
                for i in range(v):
                    if assignment[i] == c:
                        if edge_colors[(i, v)] == c:
                            ok = False
                            break
                if not ok:
                    assignment[v] = None
                    continue
                
                # Forward checking: update domains of unassigned vertices
                saved_domains = []
                conflict = False
                for j in range(v + 1, n):
                    saved_domains.append(domains[j].copy())
                    # If we assign color c to (v, new), and edge (v,j) has color c,
                    # then (j, new) cannot be color c
                    if edge_colors[(v, j)] == c:
                        if c in domains[j]:
                            domains[j].discard(c)
                            if not domains[j]:
                                conflict = True
                                break
                
                if not conflict:
                    if backtrack(v + 1):
                        return True
                
                # Undo forward checking
                if not conflict:
                    for j in range(v + 1, n):
                        domains[j] = saved_domains[j - v - 1]
                else:
                    for j in range(v + 1, n):
                        if j - v - 1 < len(saved_domains):
                            domains[j] = saved_domains[j - v - 1]
                
                assignment[v] = None
            
            return False
        
        if backtrack(0):
            return assignment[:]
        return None

    # Local search with incremental updates (refined from current)
    def local_search(N, time_limit, seed_colors=None, restarts=1):
        t_start = time.time()
        t_end = t_start + time_limit
        best_result = None
        best_bad = float('inf')

        for r in range(restarts):
            if time.time() >= t_end:
                break

            if r == 0 and seed_colors and len(seed_colors) == N * (N - 1) // 2:
                colors = np.array(seed_colors, dtype=np.int8).copy()
            else:
                colors = np.random.randint(0, 4, size=(N * (N - 1) // 2), dtype=np.int8)

            # Build adjacency matrices
            M = [np.zeros((N, N), dtype=np.int8) for _ in range(4)]
            C = np.zeros((N, N), dtype=np.int8)
            idx = 0
            for i in range(N):
                for j in range(i + 1, N):
                    c = colors[idx]
                    M[c][i, j] = 1
                    M[c][j, i] = 1
                    C[i, j] = c
                    C[j, i] = c
                    idx += 1

            M2 = [M[c] @ M[c] for c in range(4)]
            
            def count_bad():
                total = 0
                for i in range(N):
                    for j in range(i + 1, N):
                        c = C[i, j]
                        if M2[c][i, j] > 0:
                            total += M2[c][i, j]
                return total

            bad_count = count_bad()

            if bad_count == 0:
                return colors.tolist()

            if bad_count < best_bad:
                best_bad = bad_count
                best_result = colors.tolist()

            current_T = 3.0
            cooling = 0.9998
            min_T = 0.01
            no_improve_steps = 0
            max_no_improve = 5000

            while time.time() < t_end:
                if bad_count == 0:
                    return colors.tolist()

                if no_improve_steps > max_no_improve:
                    for _ in range(20):
                        ei = random.randint(0, len(colors) - 1)
                        colors[ei] = random.randint(0, 3)
                    idx = 0
                    for i in range(N):
                        for j in range(i + 1, N):
                            c = colors[idx]
                            for cc in range(4):
                                M[cc][i, j] = 0
                                M[cc][j, i] = 0
                            M[c][i, j] = 1
                            M[c][j, i] = 1
                            C[i, j] = c
                            C[j, i] = c
                            idx += 1
                    M2 = [M[c] @ M[c] for c in range(4)]
                    bad_count = count_bad()
                    if bad_count < best_bad:
                        best_bad = bad_count
                        best_result = colors.tolist()
                    no_improve_steps = 0
                    current_T = 3.0
                    continue

                # Find bad edges
                bad_edges = []
                for i in range(N):
                    for j in range(i + 1, N):
                        c = C[i, j]
                        if M2[c][i, j] > 0:
                            bad_edges.append((i, j, c))

                if not bad_edges:
                    return colors.tolist()

                u, v, c_old = random.choice(bad_edges)

                best_delta = float('inf')
                best_c = -1
                for c_new in range(4):
                    if c_new == c_old:
                        continue
                    k_old = M2[c_old][u, v]
                    k_new = M2[c_new][u, v]
                    delta = k_new - k_old
                    if delta < best_delta:
                        best_delta = delta
                        best_c = c_new

                accept = False
                if best_delta < 0:
                    accept = True
                    no_improve_steps = 0
                else:
                    p = np.exp(-best_delta / max(current_T, 1e-10))
                    if random.random() < p:
                        accept = True
                    no_improve_steps += 1

                if accept:
                    C[u, v] = best_c
                    C[v, u] = best_c
                    idx_u_v = edge_idx(u, v, N)
                    colors[idx_u_v] = best_c

                    M[c_old][u, v] = 0
                    M[c_old][v, u] = 0
                    M[best_c][u, v] = 1
                    M[best_c][v, u] = 1

                    M2[c_old] = M[c_old] @ M[c_old]
                    M2[best_c] = M[best_c] @ M[best_c]

                    bad_count = count_bad()
                    if bad_count < best_bad:
                        best_bad = bad_count
                        best_result = colors.tolist()

                current_T *= cooling
                if current_T < min_T:
                    current_T = min_T

        return best_result

    # Strategy 1: CSP-based vertex addition
    n = 32
    colors_cur = seed_32[:]

    for target in range(33, 51):
        if time.time() > DEADLINE - 25:
            break
        extended = try_add_vertex_csp(n, colors_cur, DEADLINE - 25)
        if extended is not None:
            for i in range(n):
                colors_cur.append(extended[i])
            n += 1
        else:
            # Try random perturbations
            success = False
            for _ in range(50):
                if time.time() > DEADLINE - 25:
                    break
                ei = random.randint(0, len(colors_cur) - 1)
                old = colors_cur[ei]
                colors_cur[ei] = (old + 1 + random.randint(0, 2)) % 4
                extended = try_add_vertex_csp(n, colors_cur, DEADLINE - 25)
                if extended is not None:
                    for i in range(n):
                        colors_cur.append(extended[i])
                    n += 1
                    success = True
                    break
                else:
                    colors_cur[ei] = old
            if not success:
                break

    # If we have a valid coloring, verify and try to extend more
    if n >= 40 and time.time() < DEADLINE - 20:
        M = [np.zeros((n, n), dtype=np.int8) for _ in range(4)]
        C = np.zeros((n, n), dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors_cur[idx]
                M[c][i, j] = 1
                M[c][j, i] = 1
                C[i, j] = c
                C[j, i] = c
                idx += 1
        M2 = [M[c] @ M[c] for c in range(4)]
        bad = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = C[i, j]
                if M2[c][i, j] > 0:
                    bad += M2[c][i, j]
        if bad == 0:
            for target in range(n + 1, 51):
                if time.time() > DEADLINE - 15:
                    break
                extended = try_add_vertex_csp(n, colors_cur, DEADLINE - 15)
                if extended is not None:
                    for i in range(n):
                        colors_cur.append(extended[i])
                    n += 1
                else:
                    break

    # Strategy 2: Local search for larger n
    time_remaining = DEADLINE - time.time()
    if time_remaining > 30:
        attempts = min(6, int((time_remaining - 5) / 10))
        for i in range(attempts):
            if time.time() > DEADLINE - 10:
                break
            tlim = min(12.0, (DEADLINE - 10 - time.time()))
            if tlim < 3:
                break
            res = local_search(50, tlim, seed_colors=seed_32 if i == 0 else None, restarts=1)
            if res is not None:
                M = [np.zeros((50, 50), dtype=np.int8) for _ in range(4)]
                C = np.zeros((50, 50), dtype=np.int8)
                idx = 0
                for ii in range(50):
                    for jj in range(ii + 1, 50):
                        c = res[idx]
                        M[c][ii, jj] = 1
                        M[c][jj, ii] = 1
                        C[ii, jj] = c
                        C[jj, ii] = c
                        idx += 1
                M2 = [M[c] @ M[c] for c in range(4)]
                bad = 0
                for ii in range(50):
                    for jj in range(ii + 1, 50):
                        c = C[ii, jj]
                        if M2[c][ii, jj] > 0:
                            bad += M2[c][ii, jj]
                if bad == 0:
                    return {'n': 50, 'colors': res}

    # Strategy 3: Try N=48
    time_remaining = DEADLINE - time.time()
    if time_remaining > 20:
        for i in range(3):
            if time.time() > DEADLINE - 8:
                break
            tlim = min(10.0, (DEADLINE - 8 - time.time()))
            if tlim < 3:
                break
            res = local_search(48, tlim, seed_colors=seed_32 if i == 0 else None, restarts=1)
            if res is not None:
                M = [np.zeros((48, 48), dtype=np.int8) for _ in range(4)]
                C = np.zeros((48, 48), dtype=np.int8)
                idx = 0
                for ii in range(48):
                    for jj in range(ii + 1, 48):
                        c = res[idx]
                        M[c][ii, jj] = 1
                        M[c][jj, ii] = 1
                        C[ii, jj] = c
                        C[jj, ii] = c
                        idx += 1
                M2 = [M[c] @ M[c] for c in range(4)]
                bad = 0
                for ii in range(48):
                    for jj in range(ii + 1, 48):
                        c = C[ii, jj]
                        if M2[c][ii, jj] > 0:
                            bad += M2[c][ii, jj]
                if bad == 0:
                    return {'n': 48, 'colors': res}

    # Strategy 4: Try N=46
    time_remaining = DEADLINE - time.time()
    if time_remaining > 12:
        res = local_search(46, min(8.0, time_remaining - 5), seed_colors=seed_32, restarts=1)
        if res is not None:
            M = [np.zeros((46, 46), dtype=np.int8) for _ in range(4)]
            C = np.zeros((46, 46), dtype=np.int8)
            idx = 0
            for ii in range(46):
                for jj in range(ii + 1, 46):
                    c = res[idx]
                    M[c][ii, jj] = 1
                    M[c][jj, ii] = 1
                    C[ii, jj] = c
                    C[jj, ii] = c
                    idx += 1
            M2 = [M[c] @ M[c] for c in range(4)]
            bad = 0
            for ii in range(46):
                for jj in range(ii + 1, 46):
                    c = C[ii, jj]
                    if M2[c][ii, jj] > 0:
                        bad += M2[c][ii, jj]
            if bad == 0:
                return {'n': 46, 'colors': res}

    # Fallback
    if n > 32 and len(colors_cur) == n * (n - 1) // 2:
        M = [np.zeros((n, n), dtype=np.int8) for _ in range(4)]
        C = np.zeros((n, n), dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors_cur[idx]
                M[c][i, j] = 1
                M[c][j, i] = 1
                C[i, j] = c
                C[j, i] = c
                idx += 1
        M2 = [M[c] @ M[c] for c in range(4)]
        bad = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = C[i, j]
                if M2[c][i, j] > 0:
                    bad += M2[c][i, j]
        if bad == 0:
            return {'n': n, 'colors': colors_cur}

    return {'n': 32, 'colors': seed_32}
