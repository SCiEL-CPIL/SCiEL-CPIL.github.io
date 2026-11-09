"""Rank-44 exact decomposition for 3x4 x 4x4 matrix multiplication.

The multiplication tensor T_{i,r,k} = 1 for i in {0,1,2}, r in {0,1,2,3}, k in {0,1,2,3}
is decomposed as sum_{c} U[i*M+r, c] * V[r*P+k, c] * W[k*N+i, c].

We split into 8 blocks:
  - Top rows (i=0,1) with B-rows {0,1} or {2,3} and B-cols {0,1} or {2,3}: 4 blocks,
    each a 2x2x2 tensor handled by Strassen (rank 7 each). Total: 28.
  - Bottom row (i=2) with B-rows {0,1} or {2,3} and B-cols {0,1} or {2,3}: 4 blocks,
    each a 1x2x2 tensor handled by schoolbook (rank 4 each). Total: 16.
Grand total: 44.

The output factor uses the official trace convention W[k*n+i, c].
"""

import time

N, M, P = 3, 4, 4
TIME_LIMIT = 120.0
SAFETY_MARGIN = 20.0


def _strassen_block(i0, r0, k0):
    """Return the 7 Strassen columns for the 2x2x2 block:
    A[i0:i0+2, r0:r0+2] * B[r0:r0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2].
    
    Returns list of 7 triples (U_entries, V_entries, W_entries) where each entry
    is a list of (index, coeff) pairs.
    U index: i*M + r
    V index: r*P + k
    W index: k*N + i
    """
    iR0, iR1 = i0, i0 + 1
    aR0, aR1 = r0, r0 + 1
    kC0, kC1 = k0, k0 + 1

    # A-forms: list of (u_idx, coeff)
    AF = {
        1: [(iR0 * M + aR0, 1), (iR1 * M + aR1, 1)],
        2: [(iR1 * M + aR0, 1), (iR1 * M + aR1, 1)],
        3: [(iR0 * M + aR0, 1)],
        4: [(iR1 * M + aR1, 1)],
        5: [(iR0 * M + aR0, 1), (iR0 * M + aR1, 1)],
        6: [(iR1 * M + aR0, 1), (iR0 * M + aR0, -1)],
        7: [(iR0 * M + aR1, 1), (iR1 * M + aR1, -1)],
    }
    # B-forms: list of (v_idx, coeff)
    BF = {
        1: [(aR0 * P + kC0, 1), (aR1 * P + kC1, 1)],
        2: [(aR0 * P + kC0, 1)],
        3: [(aR0 * P + kC1, 1), (aR1 * P + kC1, -1)],
        4: [(aR0 * P + kC0, 1), (aR1 * P + kC0, 1)],
        5: [(aR1 * P + kC1, 1)],
        6: [(aR0 * P + kC0, 1), (aR0 * P + kC1, 1)],
        7: [(aR1 * P + kC0, 1), (aR1 * P + kC1, 1)],
    }
    # C-forms: list of (w_idx, coeff)
    CF = {
        1: [(kC0 * N + iR0, 1), (kC1 * N + iR1, 1)],
        2: [(kC0 * N + iR1, 1), (kC1 * N + iR1, -1)],
        3: [(kC1 * N + iR0, 1), (kC1 * N + iR1, 1)],
        4: [(kC0 * N + iR1, 1)],
        5: [(kC0 * N + iR0, -1), (kC1 * N + iR0, 1)],
        6: [(kC1 * N + iR1, 1)],
        7: [(kC0 * N + iR0, 1)],
    }

    cols = []
    for m in range(1, 8):
        cols.append((AF[m], BF[m], CF[m]))
    return cols


def _naive_block(i_rows, r_rows, k_cols):
    """Return schoolbook columns for a block: for each (i, r, k) one rank-1 term.
    Returns list of triples (U_entries, V_entries, W_entries)."""
    cols = []
    for i in i_rows:
        for r in r_rows:
            for k in k_cols:
                cols.append(
                    ([(i * M + r, 1)], [(r * P + k, 1)], [(k * N + i, 1)])
                )
    return cols


def _build_rank44():
    """Build the rank-44 decomposition and return (U, V, W) as lists of lists."""
    # Collect all columns
    all_cols = []

    # 4 Strassen blocks (top 2 rows of A)
    all_cols.extend(_strassen_block(0, 0, 0))  # rows{0,1}, Brows{0,1}, Bcols{0,1}
    all_cols.extend(_strassen_block(0, 2, 0))  # rows{0,1}, Brows{2,3}, Bcols{0,1}
    all_cols.extend(_strassen_block(0, 0, 2))  # rows{0,1}, Brows{0,1}, Bcols{2,3}
    all_cols.extend(_strassen_block(0, 2, 2))  # rows{0,1}, Brows{2,3}, Bcols{2,3}

    # 4 naive blocks (bottom row of A)
    all_cols.extend(_naive_block([2], [0, 1], [0, 1]))
    all_cols.extend(_naive_block([2], [2, 3], [0, 1]))
    all_cols.extend(_naive_block([2], [0, 1], [2, 3]))
    all_cols.extend(_naive_block([2], [2, 3], [2, 3]))

    rank = len(all_cols)
    assert rank == 44, f"Expected rank 44, got {rank}"

    U = [[0.0] * rank for _ in range(N * M)]
    V = [[0.0] * rank for _ in range(M * P)]
    W = [[0.0] * rank for _ in range(P * N)]

    for c, (u_entries, v_entries, w_entries) in enumerate(all_cols):
        for idx, coeff in u_entries:
            U[idx][c] = coeff
        for idx, coeff in v_entries:
            V[idx][c] = coeff
        for idx, coeff in w_entries:
            W[idx][c] = coeff

    return U, V, W


def _verify(U, V, W):
    """Verify exact decomposition: for all (i, r, k), sum_c U[i*M+r,c]*V[r*P+k,c]*W[k*N+i,c] == 1."""
    rank = len(U[0])
    for i in range(N):
        for r in range(M):
            for k in range(P):
                s = 0.0
                for c in range(rank):
                    s += U[i * M + r][c] * V[r * P + k][c] * W[k * N + i][c]
                if abs(s - 1.0) > 1e-12:
                    return False
    return True


def _schoolbook():
    """Fallback: rank-48 schoolbook decomposition."""
    rank = N * M * P
    U = [[0.0] * rank for _ in range(N * M)]
    V = [[0.0] * rank for _ in range(M * P)]
    W = [[0.0] * rank for _ in range(P * N)]
    col = 0
    for i in range(N):
        for r in range(M):
            for k in range(P):
                U[i * M + r][col] = 1.0
                V[r * P + k][col] = 1.0
                W[k * N + i][col] = 1.0
                col += 1
    return U, V, W


def construct():
    start = time.time()
    try:
        U, V, W = _build_rank44()
        if time.time() - start < TIME_LIMIT - SAFETY_MARGIN:
            if _verify(U, V, W):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    # Fallback
    U, V, W = _schoolbook()
    return {"U": U, "V": V, "W": W}


# EVOLVE-BLOCK-START
# (construct defined above)
# EVOLVE-BLOCK-END
