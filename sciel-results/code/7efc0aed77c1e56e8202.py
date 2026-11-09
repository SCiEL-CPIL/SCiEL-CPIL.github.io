"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

# ---- Strassen rank-7 decomposition for 2x2x2 (exact integer) ----
# Indices: U row = i*2+j, V row = j*2+k, W row = k*2+i
# M1 = (A00+A11)*(B00+B11)
# M2 = (A10+A11)*B00
# M3 = A00*(B01-B11)
# M4 = A11*(B10-B00)
# M5 = (A00+A10)*B11
# M6 = (A11-A10)*(B00+B10)
# M7 = (A01-A11)*(B01+B11)
# C00 = M1 + M4 - M5 + M7
# C01 = M3 + M5
# C10 = M2 + M4
# C11 = M1 - M2 + M3 + M6
_STRASSEN_U = [
    # r: 0,1,2,3,4,5,6
    [1, 0, 1, 0, 1, 0, 0],   # A00
    [0, 1, 0, 0, 0, 1, 0],   # A10
    [0, 0, 0, 0, 0, 0, 1],   # A01
    [1, 1, 0, 1, 0, -1, -1], # A11
]
_STRASSEN_V = [
    [1, 1, 0, -1, 0, 1, 0],  # B00
    [0, 0, 1, 0, 1, 0, 1],   # B01
    [1, 0, -1, -1, 0, 1, 0], # B10
    [0, 0, 0, 0, -1, 0, 1],  # B11
]
_STRASSEN_W = [
    [1, 0, 0, 1, 0, 0, 1],   # C00
    [0, 0, 1, 0, 1, 0, 0],   # C01
    [0, 1, 0, 0, 0, 0, 0],   # C10
    [1, -1, 1, 0, 0, 1, 0],  # C11
]


def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]
    Ut = U.reshape(n, m, rank)
    Vt = V.reshape(m, p, rank)
    Wt = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', Ut, Vt, Wt)
    target = np.ones((n, m, p))
    return np.max(np.abs(T - target)) < tol


def _make_strassen_block():
    U = np.array(_STRASSEN_U, dtype=np.int64)
    V = np.array(_STRASSEN_V, dtype=np.int64)
    W = np.array(_STRASSEN_W, dtype=np.int64)
    return U, V, W


def _build_rank36(n, m, p):
    # A (2x4), B (4x5), C (2x5)
    # Partition B columns into (2,2,1): k0 = {0,1}, k1 = {2,3}, k2 = {4}
    # Four 2x2x2 blocks (i in {0,1}, j in {0,1,2,3} in pairs, k in {0,1} or {2,3})
    # Block (a,b) where a in {0,1} selects j-pair (2a,2a+1), b in {0,1} selects k-pair (2b,2b+1)
    rank = 4 * 7 + 8  # 28 + 8
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    Su, Sv, Sw = _make_strassen_block()
    col = 0
    for a in range(2):        # j-pair index
        for b in range(2):    # k-pair index
            j0 = 2 * a
            k0 = 2 * b
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i * m + (j0 + j), col] = Su[i * 2 + j, r]
                for j in range(2):
                    for k in range(2):
                        V[(j0 + j) * p + (k0 + k), col] = Sv[j * 2 + k, r]
                for k in range(2):
                    for i in range(2):
                        W[(k0 + k) * n + i, col] = Sw[k * 2 + i, r]
                col += 1

    # Remaining 2x4x1 slice: j in {0,1,2,3}, k = 4
    for i in range(2):
        for j in range(4):
            U[i * m + j, col] = 1
            V[j * p + 4, col] = 1
            W[4 * n + i, col] = 1
            col += 1

    assert col == rank, (col, rank)
    return U, V, W, rank


def _try_improve_slice(n, m, p, deadline):
    """
    Attempt to find a rank-7 decomposition for the 2x4x1 slice
    (U: 8x7, V: 20x7, W: 10x7) with exact integer entries.
    Target: for all i in {0,1}, j in {0..3}, k=4:
        sum_r U[i,j,r] V[j,4,r] W[4,i,r] = 1
    and all other (i,j,k) with k != 4 must be 0 (i.e. V[j,k,r]=0 for k != 4,
    and W[k,i,r]=0 for k != 4). So effectively V only has nonzero rows at k=4,
    W only at k=4. This reduces to: for all i,j:
        sum_r U[i,j,r] * V[j,4,r] * W[4,i,r] = 1
    with U (2x4x7), V4 (4x7), W4 (2x7).
    """
    # V4: 4x7, W4: 2x7, U: 2x4x7
    # We need sum_r U[i,j,r] V4[j,r] W4[i,r] = 1 for all i,j.
    # Search over small integer values.
    # This is a 3-way tensor of rank 7 for a 2x4x1 "all-ones" tensor.
    # The all-ones 2x4x1 tensor has rank 1 actually! U[i,j,r]=1, V4[j,r]=1, W4[i,r]=1
    # gives sum_r 1*1*1 = 7, not 1. We can scale: U=1, V4=1, W4=1/7 -> not integer.
    # For rank 1 with integers: U[i,j,0]=1, V4[j,0]=1, W4[i,0]=1 -> product sum = 1*1*1 = 1. YES!
    # So rank 1 works for the 2x4x1 all-ones slice!
    # Wait: U[i*m+j, 0] = 1 for all i,j; V[j*p+4, 0] = 1 for all j; W[4*n+i, 0] = 1 for all i.
    # Then sum_r = 1*1*1 = 1. 
    # So the slice can be done in rank 1, not 8!
    # Let's verify this doesn't interfere with the Strassen blocks.
    # The Strassen blocks use k in {0,1,2,3} for V and W. The slice uses k=4.
    # They use disjoint rows of V and W, so no interference.
    # Thus total rank = 28 + 1 = 29.
    return True  # Signal that rank 1 works for the slice


def _build_rank29(n, m, p):
    """
    Four 2x2x2 Strassen blocks (rank 28) + one rank-1 slice for 2x4x1.
    """
    rank = 4 * 7 + 1  # 29
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    Su, Sv, Sw = _make_strassen_block()
    col = 0
    for a in range(2):
        for b in range(2):
            j0 = 2 * a
            k0 = 2 * b
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i * m + (j0 + j), col] = Su[i * 2 + j, r]
                for j in range(2):
                    for k in range(2):
                        V[(j0 + j) * p + (k0 + k), col] = Sv[j * 2 + k, r]
                for k in range(2):
                    for i in range(2):
                        W[(k0 + k) * n + i, col] = Sw[k * 2 + i, r]
                col += 1

    # Rank-1 slice: k=4
    for i in range(2):
        for j in range(4):
            U[i * m + j, col] = 1
        V[j * p + 4, col] = 1
    for i in range(2):
        W[4 * n + i, col] = 1
    col += 1

    assert col == rank, (col, rank)
    return U, V, W, rank


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 130.0

    # Build rank-29 candidate
    U29, V29, W29, rank29 = _build_rank29(n, m, p)

    # Verify with exact integer arithmetic
    def _verify_exact(U, V, W, n, m, p):
        Ut = U.reshape(n, m, rank29)
        Vt = V.reshape(m, p, rank29)
        Wt = W.reshape(p, n, rank29)
        # T[i,j,k] = sum_r Ut[i,j,r] * Vt[j,k,r] * Wt[k,i,r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank29):
                        s += int(Ut[i, j, r]) * int(Vt[j, k, r]) * int(Wt[k, i, r])
                    if s != 1:
                        return False
        return True

    if _verify_exact(U29, V29, W29, n, m, p):
        best_U, best_V, best_W = U29, V29, W29
        best_rank = rank29
    else:
        # Fallback to rank 36
        U36, V36, W36, rank36 = _build_rank36(n, m, p)
        best_U, best_V, best_W = U36, V36, W36
        best_rank = rank36

    # Convert to lists
    U_list = [[int(x) for x in row] for row in best_U]
    V_list = [[int(x) for x in row] for row in best_V]
    W_list = [[int(x) for x in row] for row in best_W]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
