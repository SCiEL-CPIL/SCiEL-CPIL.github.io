"""
Rank-minimal exact tensor decomposition for 4x5 * 5x6 -> 4x6 matrix multiplication.

Strategy:
  1. Build a 2-level Strassen decomposition for the 4x4x4 core (rank 49).
  2. For the remaining parts, use schoolbook (rank 16 + 32 + 8 = 56).
  3. Combine to get total rank 105 (vs schoolbook 120).
  4. Verify correctness; fall back to schoolbook if verification fails.
  5. Try to improve the 4x4x4 core rank via local search on the decomposition
     (pruning redundant/linearly-dependent rank-1 tensors).
"""

DIMENSIONS = [4, 5, 6]
import time
import numpy as np

# EVOLVE-BLOCK-START


def _strassen_2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2.
    U: (4, 7) rows (i,j), V: (4, 7) rows (j,k), W: (4, 7) rows (k,i)
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]

    # M1 = (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # M2 = (a10+a11)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1

    # M3 = a00(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1

    # M4 = a11(b10-b00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1

    # M5 = (a00+a01)b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[3][4] = 1

    # M6 = (a10-a00)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # M7 = (a01-a11)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[1][6] = 1

    return U, V, W


def _build_4x4x4_strassen():
    """
    2-level Strassen for 4x4 * 4x4 -> 4x4. Rank = 49.
    U: (16, 49), V: (16, 49), W: (16, 49)
    Row indexing: U[i*4+j], V[j*4+k], W[k*4+i]
    """
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]

    U2, V2, W2 = _strassen_2x2()

    # Outer Strassen: 7 block-level products
    # Each block is 2x2
    # outer_U[s] = list of (block_row, block_col, coef)
    outer_U = [
        [(0, 0, 1), (1, 1, 1)],   # M1: (A00+A11)
        [(1, 0, 1), (1, 1, 1)],   # M2: (A10+A11)
        [(0, 0, 1)],               # M3: A00
        [(1, 1, 1)],               # M4: A11
        [(0, 0, 1), (0, 1, 1)],   # M5: (A00+A01)
        [(1, 0, 1), (0, 0, -1)],  # M6: (A10-A00)
        [(0, 1, 1), (1, 1, -1)],  # M7: (A01-A11)
    ]
    outer_V = [
        [(0, 0, 1), (1, 1, 1)],   # (B00+B11)
        [(0, 0, 1)],               # B00
        [(0, 1, 1), (1, 1, -1)],  # (B01-B11)
        [(1, 0, 1), (0, 0, -1)],  # (B10-B00)
        [(1, 1, 1)],               # B11
        [(0, 0, 1), (0, 1, 1)],   # (B00+B01)
        [(1, 0, 1), (1, 1, 1)],   # (B10+B11)
    ]
    outer_W = [
        [(0, 0, 1), (1, 1, 1)],   # (C00+C11)
        [(1, 0, 1), (1, 1, -1)],  # (C10-C11)
        [(0, 1, 1), (1, 1, 1)],   # (C01+C11)
        [(1, 0, 1), (0, 0, -1)],  # (C10-C00)
        [(0, 0, -1), (0, 1, 1)],  # (C01-C00)
        [(1, 1, 1)],               # C11
        [(0, 0, 1)],               # C00
    ]

    for s in range(7):
        for t in range(7):
            col = s * 7 + t

            for (bi, bj, coef) in outer_U[s]:
                for i in range(2):
                    for j in range(2):
                        gi = bi * 2 + i
                        gj = bj * 2 + j
                        a_loc = i * 2 + j
                        inner_val = U2[a_loc][t]
                        if inner_val != 0:
                            U[gi * 4 + gj][col] += inner_val * coef

            for (bj, bk, coef) in outer_V[s]:
                for j in range(2):
                    for k in range(2):
                        gj = bj * 2 + j
                        gk = bk * 2 + k
                        b_loc = j * 2 + k
                        inner_val = V2[b_loc][t]
                        if inner_val != 0:
                            V[gj * 4 + gk][col] += inner_val * coef

            for (bk, bi, coef) in outer_W[s]:
                for k in range(2):
                    for i in range(2):
                        gk = bk * 2 + k
                        gi = bi * 2 + i
                        c_loc = k * 2 + i
                        inner_val = W2[c_loc][t]
                        if inner_val != 0:
                            W[gk * 4 + gi][col] += inner_val * coef

    return U, V, W


def _build_4x4x2_schoolbook():
    """4x4 * 4x2 -> 4x2, schoolbook rank 32."""
    U = [[0] * 32 for _ in range(16)]
    V = [[0] * 32 for _ in range(8)]
    W = [[0] * 32 for _ in range(8)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U[i * 4 + j][col] = 1
                V[j * 2 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
    return U, V, W


def _build_4x1x4_outer():
    """4x1 * 1x4 -> 4x4, rank 16."""
    U = [[0] * 16 for _ in range(4)]
    V = [[0] * 16 for _ in range(4)]
    W = [[0] * 16 for _ in range(16)]
    col = 0
    for i in range(4):
        for k in range(4):
            U[i][col] = 1
            V[k][col] = 1
            W[k * 4 + i][col] = 1
            col += 1
    return U, V, W


def _build_4x1x2_outer():
    """4x1 * 1x2 -> 4x2, rank 8."""
    U = [[0] * 8 for _ in range(4)]
    V = [[0] * 8 for _ in range(2)]
    W = [[0] * 8 for _ in range(8)]
    col = 0
    for i in range(4):
        for k in range(2):
            U[i][col] = 1
            V[k][col] = 1
            W[k * 4 + i][col] = 1
            col += 1
    return U, V, W


def _combine_parts(parts, n, m, p):
    """
    Combine decomposition parts.
    Each part: (U, V, W, (i0, j0, ii, jj, k0, kk))
    U shape (ii*jj, r), V shape (jj*kk, r), W shape (kk*ii, r)
    Global: U row = (i0+i)*m + (j0+j), V row = (j0+j)*p + (k0+k), W row = (k0+k)*n + (i0+i)
    """
    total_rank = sum(len(part[0][0]) for part in parts)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    offset = 0
    for (Us, Vs, Ws, (i0, j0, ii, jj, k0, kk)) in parts:
        r = len(Us[0])
        for ri in range(ii * jj):
            i = ri // jj
            j = ri % jj
            for c in range(r):
                U[(i0 + i) * m + (j0 + j)][offset + c] = Us[ri][c]
        for ri in range(jj * kk):
            j = ri // kk
            k = ri % kk
            for c in range(r):
                V[(j0 + j) * p + (k0 + k)][offset + c] = Vs[ri][c]
        for ri in range(kk * ii):
            k = ri // ii
            i = ri % ii
            for c in range(r):
                W[(k0 + k) * n + (i0 + i)][offset + c] = Ws[ri][c]
        offset += r

    return U, V, W


def _verify_decomp(U, V, W, n, m, p):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i]))
                if abs(s - 1.0) > 1e-6:
                    return False
    return True


def _try_prune_core(U, V, W, n, m, p, deadline, max_iterations=50):
    """
    Try to reduce rank of the 4x4x4 core by identifying rank-1 tensors that
    can be removed or merged. This is a heuristic local search.
    Returns (new_U_core, new_V_core, new_W_core) with possibly lower rank.
    """
    # The core is 4x4x4, indices 0..3 for all dims
    # Build the full 4x4x4x... tensor and check which rank-1 terms are redundant
    # This is expensive; use numpy for speed
    rank = len(U[0])
    if rank <= 40:
        return U, V, W  # unlikely to improve much

    # Check linear dependence: for each column c, see if the rank-1 tensor
    # (U[:,c], V[:,c], W[:,c]) is in the span of others.
    # Simpler: try removing each column and re-verify.
    # With rank 49, removing one means rank 48. Check if still valid.
    # This is O(rank * n*m*p*rank) which is 49*64*49 ≈ 154k ops per check, fine.

    best_rank = rank
    best_U, best_V, best_W = U, V, W
    active = list(range(rank))

    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        found_removal = False
        for idx in range(len(active)):
            c = active[idx]
            # Build without this column
            remaining = active[:idx] + active[idx + 1:]
            if len(remaining) >= best_rank:
                continue
            # Quick check: verify core decomposition without column c
            valid = True
            for i in range(4):
                if not valid:
                    break
                for j in range(4):
                    if not valid:
                        break
                    for k in range(4):
                        s = 0.0
                        for cc in remaining:
                            s += U[i * 4 + j][cc] * V[j * 4 + k][cc] * W[k * 4 + i][cc]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
            if valid:
                active = remaining
                best_rank = len(remaining)
                found_removal = True
                break
        if not found_removal:
            break

    if best_rank == rank:
        return U, V, W

    # Rebuild with only active columns
    new_U = [[U[r][c] for c in active] for r in range(len(U))]
    new_V = [[V[r][c] for c in active] for r in range(len(V))]
    new_W = [[W[r][c] for c in active] for r in range(len(W))]
    return new_U, new_V, new_W


def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s limit
    n, m, p = DIMENSIONS  # 4, 5, 6

    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p  # 120
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = schoolbook()
    best_rank = n * m * p

    try:
        # Build the 4x4x4 core using 2-level Strassen (rank 49)
        U_core, V_core, W_core = _build_4x4x4_strassen()

        # Attempt to prune the core
        U_core, V_core, W_core = _try_prune_core(U_core, V_core, W_core, n, m, p, deadline)
        core_rank = len(U_core[0])

        # Build cross terms
        U_cross1, V_cross1, W_cross1 = _build_4x1x4_outer()     # rank 16
        U_cross2, V_cross2, W_cross2 = _build_4x4x2_schoolbook()  # rank 32
        U_cross3, V_cross3, W_cross3 = _build_4x1x2_outer()     # rank 8

        parts = [
            (U_core, V_core, W_core, (0, 0, 4, 4, 0, 4)),
            (U_cross1, V_cross1, W_cross1, (0, 4, 4, 1, 0, 4)),
            (U_cross2, V_cross2, W_cross2, (0, 0, 4, 4, 4, 2)),
            (U_cross3, V_cross3, W_cross3, (0, 4, 4, 1, 4, 2)),
        ]

        U, V, W = _combine_parts(parts, n, m, p)
        total_rank = len(U[0])

        if _verify_decomp(U, V, W, n, m, p):
            if total_rank < best_rank:
                best_result = {"U": U, "V": V, "W": W}
                best_rank = total_rank

        if time.time() > deadline:
            return best_result

    except Exception:
        pass

    return best_result

# EVOLVE-BLOCK-END
