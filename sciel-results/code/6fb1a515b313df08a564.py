"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 5, 5

    # ---- Strassen 2x2x2 block (7 rank terms) ----
    def add_strassen_block(U, V, W, i0, j0, k0, col_offset):
        # U index: i*m + j ; V index: j*p + k ; W index: k*n + i
        # M1 = (A00+A11)(B00+B11)
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        # M2 = (A10+A11)B00
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        # M3 = A00(B01-B11)
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        # M4 = A11(B10-B00)
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        # M5 = (A00+A01)B11
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        # M6 = (A10-A00)(B00+B01)
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        # M7 = (A01-A11)(B10+B11)
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    # ---- Laderman 3x3x3 block (7 rank terms) ----
    def add_laderman_block(U, V, W, i0, j0, k0, col_offset):
        # M1 = (A00+A11)(B00+B11)
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        # M2 = (A10+A20)(B00+B01)
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+2)*m + j0][c] += 1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        W[k0*n + (i0+2)][c] += 1
        W[(k0+1)*n + (i0+2)][c] += 1
        # M3 = (A00+A01)(B10+B11)
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        # M4 = A20(B01-B11)
        c = col_offset + 3
        U[(i0+2)*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + (i0+2)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        # M5 = A02(B20-B21)
        c = col_offset + 4
        U[i0*m + (j0+2)][c] += 1
        V[(j0+2)*p + k0][c] += 1
        V[(j0+2)*p + (k0+1)][c] += -1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + i0][c] += 1
        # M6 = A11(B21-B00)
        c = col_offset + 5
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+2)*p + (k0+1)][c] += 1
        V[j0*p + k0][c] += -1
        W[(k0+1)*n + (i0+1)][c] += 1
        W[k0*n + (i0+1)][c] += 1
        # M7 = A22(B00-B10)
        c = col_offset + 6
        U[(i0+2)*m + (j0+2)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + k0][c] += -1
        W[k0*n + (i0+2)][c] += 1
        W[k0*n + (i0+1)][c] += 1

    # Build the full target tensor once (for fast verification)
    target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1

    def build_decomposition(blocks):
        """Build (U,V,W,rank) from a list of (type,i0,j0,k0) blocks + schoolbook fill."""
        # coverage
        covered = [[[False]*p for _ in range(m)] for _ in range(n)]
        total_rank = 0
        for btype, i0, j0, k0 in blocks:
            size = 2 if btype == 'strassen' else 3
            for di in range(size):
                for dj in range(size):
                    for dk in range(size):
                        covered[i0+di][j0+dj][k0+dk] = True
            total_rank += 7
        remaining = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered[i][j][k]:
                        remaining += 1
        total_rank += remaining

        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]

        col = 0
        for btype, i0, j0, k0 in blocks:
            if btype == 'strassen':
                add_strassen_block(U, V, W, i0, j0, k0, col)
            else:
                add_laderman_block(U, V, W, i0, j0, k0, col)
            col += 7
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered[i][j][k]:
                        U[i*m+j][col] += 1
                        V[j*p+k][col] += 1
                        W[k*n+i][col] += 1
                        col += 1
        return U, V, W, total_rank

    def verify(U, V, W):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        T_decomp = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        return np.allclose(T_decomp, target, atol=1e-10)

    def is_valid_placement(blocks):
        covered = [[[False]*p for _ in range(m)] for _ in range(n)]
        for btype, i0, j0, k0 in blocks:
            size = 2 if btype == 'strassen' else 3
            if i0 < 0 or i0+size > n or j0 < 0 or j0+size > m or k0 < 0 or k0+size > p:
                return False
            for di in range(size):
                for dj in range(size):
                    for dk in range(size):
                        if covered[i0+di][j0+dj][k0+dk]:
                            return False
                        covered[i0+di][j0+dj][k0+dk] = True
        return True

    # ---- Fast guaranteed baseline: 4 non-overlapping Strassen blocks (rank 71) ----
    base_config = [
        ('strassen', 0, 0, 0),
        ('strassen', 0, 2, 0),
        ('strassen', 0, 0, 2),
        ('strassen', 0, 2, 2),
    ]
    U, V, W, base_rank = build_decomposition(base_config)
    best_result = {"U": U, "V": V, "W": W}
    best_rank = base_rank
    # Verify baseline once (should pass; if not, fall back to schoolbook)
    if not verify(U, V, W):
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        c2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][c2] = 1
                    V_s[j*p+k][c2] = 1
                    W_s[k*n+i][c2] = 1
                    c2 += 1
        return {"U": U_s, "V": V_s, "W": W_s}

    # ---- Candidate positions ----
    strassen_positions = [('strassen', i0, j0, k0) for i0 in range(n-1)
                          for j0 in range(m-1) for k0 in range(p-1)]
    laderman_positions = [('larderman', i0, j0, k0) for i0 in range(n-2)
                          for j0 in range(m-2) for k0 in range(p-2)]

    def rank_estimate(blocks):
        """Cheap rank estimate (upper bound) without building."""
        covered = [[[False]*p for _ in range(m)] for _ in range(n)]
        r = 0
        for btype, i0, j0, k0 in blocks:
            size = 2 if btype == 'strassen' else 3
            for di in range(size):
                for dj in range(size):
                    for dk in range(size):
                        covered[i0+di][j0+dj][k0+dk] = True
            r += 7
        remaining = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered[i][j][k]:
                        remaining += 1
        return r + remaining

    def try_config(config):
        """Build + verify config; update best. Returns rank or None."""
        nonlocal best_rank, best_result
        if not is_valid_placement(config):
            return None
        est = rank_estimate(config)
        if est >= best_rank:
            return None  # cannot improve; skip expensive build/verify
        U, V, W, rank = build_decomposition(config)
        if verify(U, V, W):
            if rank < best_rank:
                best_rank = rank
                best_result = {"U": U, "V": V, "W": W}
            return rank
        return None

    # ---- Search phase: try to beat the rank-71 baseline ----
    # (a) Greedy from base: add a 5th non-overlapping block
    def greedy_improve(initial_config, max_iters=400):
        nonlocal best_rank, best_result
        current = list(initial_config)
        improved = True
        iters = 0
        all_positions = strassen_positions + laderman_positions
        while improved and iters < max_iters and time.time() < deadline - 12:
            improved = False
            iters += 1
            # Try adding each block
            for pos in all_positions:
                if time.time() > deadline - 12:
                    break
                new_config = current + [pos]
                if try_config(new_config) is not None and best_rank < len(current)*7 + 100:
                    # improved (best_rank updated); adopt
                    current = new_config
                    improved = True
                    break
            if improved or time.time() > deadline - 12:
                continue
            # Try removing one block
            for i in range(len(current)):
                if time.time() > deadline - 12:
                    break
                new_config = current[:i] + current[i+1:]
                if try_config(new_config) is not None:
                    current = new_config
                    improved = True
                    break
            if improved or time.time() > deadline - 12:
                continue
            # Try swap: remove one, add another
            for i in range(len(current)):
                if time.time() > deadline - 12:
                    break
                removed = current[i]
                base_no_i = current[:i] + current[i+1:]
                for pos in all_positions:
                    if time.time() > deadline - 12:
                        break
                    if pos == removed:
                        continue
                    new_config = base_no_i + [pos]
                    if try_config(new_config) is not None:
                        current = new_config
                        improved = True
                        break
                if improved:
                    break
        return current

    if time.time() < deadline - 30:
        greedy_improve(base_config, max_iters=300)

    # (b) Exhaustive: all 5-tuples of non-overlapping Strassen blocks
    if time.time() < deadline - 15:
        for combo5 in combinations(strassen_positions, 5):
            if time.time() > deadline - 8:
                break
            if rank_estimate(list(combo5)) >= best_rank:
                continue
            try_config(list(combo5))

    # (c) Laderman + Strassen mixed via greedy
    if time.time() < deadline - 20:
        for lpos in laderman_positions:
            if time.time() > deadline - 12:
                break
            greedy_improve([lpos], max_iters=80)

    # (d) 2 Laderman blocks
    if time.time() < deadline - 15:
        for l_combo in combinations(laderman_positions, 2):
            if time.time() > deadline - 8:
                break
            if rank_estimate(list(l_combo)) >= best_rank:
                continue
            try_config(list(l_combo))

    # (e) 6-tuple Strassen
    if time.time() < deadline - 12:
        for combo6 in combinations(strassen_positions, 6):
            if time.time() > deadline - 5:
                break
            if rank_estimate(list(combo6)) >= best_rank:
                continue
            try_config(list(combo6))

    # Safety: if we somehow lost the baseline, rebuild it
    if best_result is None:
        U, V, W, _ = build_decomposition(base_config)
        best_result = {"U": U, "V": W and V, "W": W}

    if time.time() > deadline - 2:
        # Emergency: return guaranteed baseline
        U, V, W, _ = build_decomposition(base_config)
        return {"U": U, "V": V, "W": W}

    return best_result
# EVOLVE-BLOCK-END
