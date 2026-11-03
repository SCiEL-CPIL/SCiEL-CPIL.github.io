"""Rank-minimal exact tensor decomposition for 4x5x5 matrix multiplication.

Strategy: build a 96-rank block-Strassen ansatz (split m=5 into 1+4, apply
Strassen on the 2x2 sub-blocks of the 4x4 part), then attempt to reduce the
rank further by identifying and eliminating linearly dependent rank-1 terms
via exact rational Gaussian elimination on the contribution matrix.
"""
import time
import numpy as np
from fractions import Fraction

_DIMS = (4, 5, 5)
_DEADLINE = 140  # seconds (150s kill, keep 10s margin)

# EVOLVE-BLOCK-START


def _strassen_2x2():
    """Strassen 7-product for 2x2x2. Returns (U, V, W) each 4x7."""
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, -1, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V = [
        [1, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, -1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, -1, 0, 0, 1, 0]
    ]
    return U, V, W


def _build_96_rank():
    """Build a 96-rank decomposition: m=5 split into 1+4; 4x4 part uses 2x2 Strassen blocks.

    Layout:
      Part 1: A[:,0] * B[0,:]  -> 4*5 = 20 rank-1 terms (schoolbook)
      Part 2: A[:,1:5] * B[1:5,:]  -> 4x4 * 4x5
        Split A into 2x2 row blocks (rows 0-1, 2-3) and B cols into 2+3:
        - A[0:2,1:3]*B[1:3,0:2]: Strassen 7
        - A[0:2,1:3]*B[1:3,2:5]: schoolbook 2*2*3=12
        - A[0:2,3:5]*B[1:3,0:2]: Strassen 7
        - A[0:2,3:5]*B[1:3,2:5]: schoolbook 12
        - A[2:4,1:3]*B[3:5,0:2]: Strassen 7
        - A[2:4,1:3]*B[3:5,2:5]: schoolbook 12
        - A[2:4,3:5]*B[3:5,0:2]: Strassen 7
        - A[2:4,3:5]*B[3:5,2:5]: schoolbook 12
        Total Part 2: 4*7 + 4*12 = 76
    Grand total: 20 + 76 = 96
    """
    n, m, p = 4, 5, 5
    total_rank = 96
    U = [[0] * total_rank for _ in range(n * m)]   # 20 rows
    V = [[0] * total_rank for _ in range(m * p)]   # 25 rows
    W = [[0] * total_rank for _ in range(p * n)]   # 20 rows
    col = 0
    sU, sV, sW = _strassen_2x2()

    # Part 1: A[:,0] * B[0,:]
    for i in range(4):
        for k in range(5):
            U[i * 5 + 0][col] = 1
            V[0 * 5 + k][col] = 1
            W[k * 4 + i][col] = 1
            col += 1
    # col == 20

    # Part 2: 4x4 * 4x5
    # A row indices in U: A[i,j] -> U[i*5+j], j in {1,2,3,4}
    #   A[0,1]=U[1], A[0,2]=U[2], A[0,3]=U[3], A[0,4]=U[4]
    #   A[1,1]=U[6], A[1,2]=U[7], A[1,3]=U[8], A[1,4]=U[9]
    #   A[2,1]=U[11], A[2,2]=U[12], A[2,3]=U[13], A[2,4]=U[14]
    #   A[3,1]=U[16], A[3,2]=U[17], A[3,3]=U[18], A[3,4]=U[19]
    # B row indices in V: B[j,k] -> V[j*5+k], j in {1,2,3,4}, k in {0..4}
    #   B[1,0]=V[5], B[1,1]=V[6], B[1,2]=V[7], B[1,3]=V[8], B[1,4]=V[9]
    #   B[2,0]=V[10], B[2,1]=V[11], B[2,2]=V[12], B[2,3]=V[13], B[2,4]=V[14]
    #   B[3,0]=V[15], B[3,1]=V[16], B[3,2]=V[17], B[3,3]=V[18], B[3,4]=V[19]
    #   B[4,0]=V[20], B[4,1]=V[21], B[4,2]=V[22], B[4,3]=V[23], B[4,4]=V[24]
    # C row indices in W: C[i,k] -> W[k*4+i]
    #   C[0,0]=W[0], C[0,1]=W[1], C[0,2]=W[2], C[0,3]=W[3], C[0,4]=W[4]
    #   C[1,0]=W[5], C[1,1]=W[6], C[1,2]=W[7], C[1,3]=W[8], C[1,4]=W[9]
    #   C[2,0]=W[10], C[2,1]=W[11], C[2,2]=W[12], C[2,3]=W[13], C[2,4]=W[14]
    #   C[3,0]=W[15], C[3,1]=W[16], C[3,2]=W[17], C[3,3]=W[18], C[3,4]=W[19]

    # Block (0,0): A[0:2,1:3] * B[1:3,0:2] -> Strassen
    a_rows = [1, 2, 6, 7]
    b_rows = [5, 6, 10, 11]
    c_rows = [0, 1, 5, 6]
    for i in range(4):
        for c in range(7):
            U[a_rows[i]][col + c] = sU[i][c]
            V[b_rows[i]][col + c] = sV[i][c]
            W[c_rows[i]][col + c] = sW[i][c]
    col += 7

    # Block (0,1): A[0:2,1:3] * B[1:3,2:5] -> schoolbook
    a_rows = [1, 2, 6, 7]
    b_rows = [7, 8, 9, 12, 13, 14]
    for ii in range(2):
        for jj in range(2):
            for kk in range(3):
                U[a_rows[ii * 2 + jj]][col] = 1
                V[b_rows[jj * 3 + kk]][col] = 1
                W[(2 + kk) * 4 + ii][col] = 1
                col += 1
    # col += 12

    # Block (0,2): A[0:2,3:5] * B[1:3,0:2] -> Strassen
    a_rows = [3, 4, 8, 9]
    b_rows = [5, 6, 10, 11]
    c_rows = [0, 1, 5, 6]
    for i in range(4):
        for c in range(7):
            U[a_rows[i]][col + c] = sU[i][c]
            V[b_rows[i]][col + c] = sV[i][c]
            W[c_rows[i]][col + c] = sW[i][c]
    col += 7

    # Block (0,3): A[0:2,3:5] * B[1:3,2:5] -> schoolbook
    a_rows = [3, 4, 8, 9]
    b_rows = [7, 8, 9, 12, 13, 14]
    for ii in range(2):
        for jj in range(2):
            for kk in range(3):
                U[a_rows[ii * 2 + jj]][col] = 1
                V[b_rows[jj * 3 + kk]][col] = 1
                W[(2 + kk) * 4 + ii][col] = 1
                col += 1
    # col += 12

    # Block (1,0): A[2:4,1:3] * B[3:5,0:2] -> Strassen
    a_rows = [11, 12, 16, 17]
    b_rows = [15, 16, 20, 21]
    c_rows = [10, 11, 15, 16]
    for i in range(4):
        for c in range(7):
            U[a_rows[i]][col + c] = sU[i][c]
            V[b_rows[i]][col + c] = sV[i][c]
            W[c_rows[i]][col + c] = sW[i][c]
    col += 7

    # Block (1,1): A[2:4,1:3] * B[3:5,2:5] -> schoolbook
    a_rows = [11, 12, 16, 17]
    b_rows = [17, 18, 19, 22, 23, 24]
    for ii in range(2):
        for jj in range(2):
            for kk in range(3):
                U[a_rows[ii * 2 + jj]][col] = 1
                V[b_rows[jj * 3 + kk]][col] = 1
                W[(2 + kk) * 4 + (2 + ii)][col] = 1
                col += 1
    # col += 12

    # Block (1,2): A[2:4,3:5] * B[3:5,0:2] -> Strassen
    a_rows = [13, 14, 18, 19]
    b_rows = [15, 16, 20, 21]
    c_rows = [10, 11, 15, 16]
    for i in range(4):
        for c in range(7):
            U[a_rows[i]][col + c] = sU[i][c]
            V[b_rows[i]][col + c] = sV[i][c]
            W[c_rows[i]][col + c] = sW[i][c]
    col += 7

    # Block (1,3): A[2:4,3:5] * B[3:5,2:5] -> schoolbook
    a_rows = [13, 14, 18, 19]
    b_rows = [17, 18, 19, 22, 23, 24]
    for ii in range(2):
        for jj in range(2):
            for kk in range(3):
                U[a_rows[ii * 2 + jj]][col] = 1
                V[b_rows[jj * 3 + kk]][col] = 1
                W[(2 + kk) * 4 + (2 + ii)][col] = 1
                col += 1
    # col += 12

    assert col == total_rank, f"Rank mismatch: expected {total_rank}, got {col}"
    return U, V, W


def _build_98_rank():
    """Alternative 98-rank decomposition: m=5 split into 2+3."""
    n, m, p = 4, 5, 5
    total_rank = 98
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    col = 0
    sU, sV, sW = _strassen_2x2()

    # A_top_left (2x2) * B_top_left (2x2) - Strassen
    a_rows = [0, 1, 5, 6]
    b_rows = [0, 1, 5, 6]
    c_rows = [0, 1, 4, 5]
    for i in range(4):
        for c in range(7):
            U[a_rows[i]][col + c] = sU[i][c]
            V[b_rows[i]][col + c] = sV[i][c]
            W[c_rows[i]][col + c] = sW[i][c]
    col += 7

    # A_top_left (2x2) * B_top_right (2x3) - schoolbook
    a_rows = [0, 1, 5, 6]
    b_rows = [2, 3, 4, 7, 8, 9]
    for ii in range(2):
        for jj in range(2):
            for kk in range(3):
                U[a_rows[ii * 2 + jj]][col] = 1
                V[b_rows[jj * 3 + kk]][col] = 1
                W[(2 + kk) * 4 + ii][col] = 1
                col += 1

    # A_bot_left (2x2) * B_top_left (2x2) - Strassen
    a_rows = [10, 11, 15, 16]
    b_rows = [0, 1, 5, 6]
    c_rows = [8, 10, 12, 14]
    for i in range(4):
        for c in range(7):
            U[a_rows[i]][col + c] = sU[i][c]
            V[b_rows[i]][col + c] = sV[i][c]
            W[c_rows[i]][col + c] = sW[i][c]
    col += 7

    # A_bot_left (2x2) * B_top_right (2x3) - schoolbook
    a_rows = [10, 11, 15, 16]
    b_rows = [2, 3, 4, 7, 8, 9]
    for ii in range(2):
        for jj in range(2):
            for kk in range(3):
                U[a_rows[ii * 2 + jj]][col] = 1
                V[b_rows[jj * 3 + kk]][col] = 1
                W[(2 + kk) * 4 + (2 + ii)][col] = 1
                col += 1

    # A_right (4x3) * B_bot (3x5) - schoolbook
    for i in range(4):
        for j in range(3):
            for k in range(5):
                U[i * 5 + (2 + j)][col] = 1
                V[(2 + j) * 5 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1

    assert col == total_rank
    return U, V, W


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition, rank = n*m*p."""
    rank = n * m * p
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
    return U, V, W


def _verify(U, V, W, n, m, p, trials=2):
    """Verify decomposition correctness with random matrices."""
    np.random.seed(12345)
    rank = len(W[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_comp = np.zeros((n, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        C_comp[i, k] += U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
        if not np.allclose(C_comp, C_expected, atol=1e-9):
            return False
    return True


def _attempt_reduction(U, V, W, n, m, p, deadline):
    """Try to reduce rank by finding linearly dependent rank-1 terms.

    Build the 100 x rank matrix A where A[i*m*p + j*p + k, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    The constraint is A @ x = 1 (all ones), where x_r is the weight of term r.
    If the system has a solution with some x_r = 0, we can drop those terms.
    We perform Gaussian elimination to find the minimum-support solution.
    """
    t0 = time.time()
    rank = len(W[0])
    num_constraints = n * m * p  # 100

    # Build A as list of Fraction rows
    A = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row = []
                for r in range(rank):
                    row.append(Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[k * n + i][r]))
                A.append(row)
    b = [Fraction(1)] * num_constraints

    # Gaussian elimination over Q
    # Work on augmented matrix [A | b]
    m_rows = len(A)
    n_cols = rank
    M = [A[i][:] + [b[i]] for i in range(m_rows)]

    pivot_cols = []
    rank_found = 0

    for col in range(n_cols):
        if time.time() - t0 > deadline - 5:
            return None

        # Find pivot
        pivot = -1
        for row in range(rank_found, m_rows):
            if M[row][col] != 0:
                pivot = row
                break
        if pivot == -1:
            continue

        if pivot != rank_found:
            M[rank_found], M[pivot] = M[pivot], M[rank_found]

        pivot_cols.append(col)

        pv = M[rank_found][col]
        # Normalize
        for j in range(col, n_cols + 1):
            M[rank_found][j] = M[rank_found][j] / pv

        # Eliminate
        for row in range(m_rows):
            if row == rank_found:
                continue
            f = M[row][col]
            if f != 0:
                for j in range(col, n_cols + 1):
                    M[row][j] = M[row][j] - f * M[rank_found][j]

        rank_found += 1

    # Check consistency
    for row in range(rank_found, m_rows):
        if all(M[row][j] == 0 for j in range(n_cols)) and M[row][n_cols] != 0:
            return None  # Inconsistent

    # Find a sparse solution: set free variables to 0, solve for pivots
    free = [c for c in range(n_cols) if c not in pivot_cols]
    x = [Fraction(0)] * n_cols
    for i in range(rank_found):
        pc = pivot_cols[i]
        x[pc] = M[i][n_cols]

    # Nonzero terms
    nz = [r for r in range(n_cols) if x[r] != 0]
    if len(nz) >= rank:
        return None

    # Build reduced decomposition
    new_rank = len(nz)
    mapping = {r: idx for idx, r in enumerate(nz)}
    new_U = [[0] * new_rank for _ in range(len(U))]
    new_V = [[0] * new_rank for _ in range(len(V))]
    new_W = [[0] * new_rank for _ in range(len(W))]
    for r in nz:
        idx = mapping[r]
        scale = x[r]
        for i in range(len(U)):
            new_U[i][idx] = U[i][r] * scale
        for j in range(len(V)):
            new_V[j][idx] = V[j][r] * scale
        for k in range(len(W)):
            new_W[k][idx] = W[k][r] * scale

    return new_U, new_V, new_W


def construct():
    """Construct the rank-minimal exact tensor decomposition for 4x5x5 matrix multiplication."""
    t0 = time.time()
    n, m, p = 4, 5, 5

    # Start with 96-rank construction
    U, V, W = _build_96_rank()
    best_rank = len(W[0])

    # Verify
    if time.time() - t0 < _DEADLINE - 20:
        if not _verify(U, V, W, n, m, p):
            U, V, W = _build_98_rank()
            best_rank = len(W[0])
            if not _verify(U, V, W, n, m, p):
                U, V, W = _schoolbook(n, m, p)
                best_rank = len(W[0])

    # Try to reduce further via linear algebra
    if time.time() - t0 < _DEADLINE - 10:
        reduced = _attempt_reduction(U, V, W, n, m, p, _DEADLINE)
        if reduced is not None:
            rU, rV, rW = reduced
            if time.time() - t0 < _DEADLINE - 5:
                if _verify(rU, rV, rW, n, m, p):
                    U, V, W = rU, rV, rW
                    best_rank = len(W[0])

    # Final deadline check
    if time.time() - t0 > _DEADLINE - 2:
        U, V, W = _schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
