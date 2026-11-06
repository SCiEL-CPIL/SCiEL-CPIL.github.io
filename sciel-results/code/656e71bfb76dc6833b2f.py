import time
import random
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0  # Safety margin

    # --- Helper: GF(16) multiplication for the Clebsch graph ---
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

    def build_seed_32():
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return colors

    def solve_limited(N, time_limit, seed_colors=None):
        t_start = time.time()
        t_end = t_start + time_limit

        if seed_colors and len(seed_colors) == N * (N - 1) // 2:
            colors = np.array(seed_colors, dtype=np.int8).copy()
        else:
            colors = np.random.randint(0, 4, size=(N * (N - 1) // 2), dtype=np.int8)

        M = [np.zeros((N, N), dtype=np.int8) for _ in range(4)]
        C = np.zeros((N, N), dtype=np.int8)
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                c = colors[idx]
                M[c][i][j] = 1
                M[c][j][i] = 1
                C[i][j] = c
                C[j][i] = c
                idx += 1

        # Compute M2 = M @ M for each color
        M2 = [M[c].astype(np.int32) @ M[c].astype(np.int32) for c in range(4)]

        # Precompute upper triangle indices for vectorized bad edge detection
        rows, cols = np.triu_indices(N, k=1)
        num_edges = len(rows)

        def get_bad_edges():
            """Vectorized bad edge detection."""
            # For each edge (i,j), check if M2[C[i,j]][i,j] > 0
            # We need to gather M2 values at (rows, cols) for each color
            # Build a combined check: for each edge, get its color and check M2
            edge_colors = C[rows, cols]  # shape (num_edges,)
            
            # For each color c, find edges of that color where M2[c] > 0
            bad_mask = np.zeros(num_edges, dtype=bool)
            for c in range(4):
                mask_c = (edge_colors == c)
                if np.any(mask_c):
                    bad_mask |= mask_c & (M2[c][rows, cols] > 0)
            
            bad_indices = np.where(bad_mask)[0]
            if len(bad_indices) == 0:
                return []
            # Return list of (u, v, color) tuples
            return [(int(rows[i]), int(cols[i]), int(edge_colors[i])) for i in bad_indices]

        bad_edges = get_bad_edges()

        if not bad_edges:
            return colors.tolist()

        current_T = 10.0
        cooling = 0.999
        min_T = 0.01

        while time.time() < t_end:
            if not bad_edges:
                return colors.tolist()

            # Pick a random bad edge
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

            # Acceptance
            accept = False
            if best_delta < 0:
                accept = True
            else:
                p = np.exp(-best_delta / current_T)
                if random.random() < p:
                    accept = True

            if accept:
                C[u, v] = best_c
                C[v, u] = best_c
                idx_u_v = u * N - u * (u + 1) // 2 + (v - u - 1)
                colors[idx_u_v] = best_c

                # Update M
                M[c_old][u, v] = 0
                M[c_old][v, u] = 0
                M[best_c][u, v] = 1
                M[best_c][v, u] = 1

                # Rank-1 update for M2
                # When removing edge (u,v) from color c_old:
                # M2[c_old][i,j] -= M[c_old][i,u]*M[c_old][u,j] + M[c_old][i,v]*M[c_old][v,j]
                # But since we're doing boolean matrices, M[c][i,k] is 0 or 1
                # M2[c][i,j] = sum_k M[c][i,k] * M[c][k,j]
                # Removing edge (u,v) from color c:
                # For each i, M2[c][i,j] changes by -M[c][i,u] (since M[c][u,j] was 1, now 0)
                # Wait, let me think more carefully.
                # M2[c][i,j] = sum_k M[c][i,k] * M[c][k,j]
                # When we set M[c][u,v] = 0 (and M[c][v,u] = 0):
                # For M2[c][i,j]: the term k=u contributes M[c][i,u]*M[c][u,j]
                # Since M[c][u,v]=1 before, M[c][u,j] was 1 for j=v.
                # After setting M[c][u,v]=0, for j=v: M2[c][i,v] loses M[c][i,u]
                # Similarly for k=v: M2[c][i,u] loses M[c][i,v]
                # And for M2[c][v,i]: similar
                # Actually, let's use the standard rank-1 update:
                # If we change M[c] by adding/removing row u and column v:
                # M2_new = M2_old - M[:,u] @ M[u,:] - M[:,v] @ M[v,:] + ... 
                # This is getting complex. Let me just recompute for the two affected colors.
                # But that's O(N^2). Let me try the rank-1 approach.
                
                # For color c_old, we remove edge (u,v):
                # M2[c_old] = (M[c_old] - e_u e_v^T - e_v e_u^T) @ (M[c_old] - e_u e_v^T - e_v e_u^T)
                # This is complex. Let me just recompute for these two colors.
                # For N=50, N^2 = 2500 operations per color, so 5000 total. Still much better than before.
                
                M2[c_old] = M[c_old].astype(np.int32) @ M[c_old].astype(np.int32)
                M2[best_c] = M[best_c].astype(np.int32) @ M[best_c].astype(np.int32)

                # Update bad_edges
                bad_edges = get_bad_edges()

            current_T *= cooling
            if current_T < min_T:
                current_T = min_T

        return None

    # Plan:
    # 1. Try N=50 with multiple restarts.
    # 2. If fail, try N=48, 46, etc.
    # 3. Fallback to Seed 32.

    seed_32 = build_seed_32()

    # Try N=50 with multiple restarts
    num_restarts_50 = 5
    time_per_restart_50 = 18.0

    for i in range(num_restarts_50):
        if time.time() > DEADLINE - 10:
            break
        res = solve_limited(50, time_per_restart_50)
        if res:
            return {'n': 50, 'colors': res}

    # Try N=48
    for i in range(3):
        if time.time() > DEADLINE - 10:
            break
        res = solve_limited(48, 12.0)
        if res:
            return {'n': 48, 'colors': res}

    # Try N=46
    for i in range(2):
        if time.time() > DEADLINE - 10:
            break
        res = solve_limited(46, 8.0)
        if res:
            return {'n': 46, 'colors': res}

    # Fallback to Seed 32
    return {'n': 32, 'colors': seed_32}
