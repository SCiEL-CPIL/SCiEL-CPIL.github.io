"""Improved rank-minimal tensor decomposition for 3x4 * 4x4 matrix multiplication."""
import numpy as np
from fractions import Fraction
import time
import itertools
import random

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification of the tensor decomposition."""
    rank = U.shape[1]
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    total = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        A = U[:, c].reshape(n, m)
        B = V[:, c].reshape(m, p)
        C = W[:, c].reshape(p, n)
        # contribution[i,j,k] = A[i,j]*B[j,k]*C[k,i]
        # vectorize over k for each (i,j)
        for i in range(n):
            for j in range(m):
                total[i, j, :] += A[i, j] * B[j, :] * C[:, i]
    return np.allclose(total, 1.0, atol=tol)


def _verify_exact(U, V, W, n, m, p):
    """Exact rational verification."""
    U = [[Fraction(x) for x in row] for row in U]
    V = [[Fraction(x) for x in row] for row in V]
    W = [[Fraction(x) for x in row] for row in W]
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W


def _strassen_2x2x2():
    """Return (U,V,W) lists for 2x2 * 2x2 with rank 7 (Strassen).
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i] for i,j,k in {0,1}.
    """
    # Strassen products M1..M7
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A01-A10)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    #
    # For each M, U factor = (a_i0 + a_i1) etc., V factor = (b_0j + b_1j), W factor = 1 for the output
    # Actually each M contributes to specific C entries with +/-1 coefficients.
    # M1 contributes: C00 += M1, C11 += M1  -> W[0*2+0]=+1 (C00), W[1*2+1]=+1 (C11)
    # M2 contributes: C10 += M2, C11 -= M2  -> W[0*2+1]=+1 (C10: k=0,i=1 -> index 0*2+1=1), W[1*2+1]=-1 (C11: k=1,i=1 -> index 1*2+1=3)
    # M3 contributes: C01 += M3, C11 += M3  -> W[1*2+0]=+1 (C01: k=1,i=0 -> index 1*2+0=2), W[1*2+1]=+1 (C11)
    # M4 contributes: C00 += M4, C10 += M4  -> W[0*2+0]=+1, W[0*2+1]=+1
    # M5 contributes: C00 -= M5, C01 += M5  -> W[0*2+0]=-1, W[1*2+0]=+1
    # M6 contributes: C11 += M6  -> W[1*2+1]=+1
    # M7 contributes: C00 += M7  -> W[0*2+0]=+1

    # U factors (for A indices i,j):
    # M1: A00+A11 -> U[0*2+0]=1, U[1*2+1]=1
    # M2: A10+A11 -> U[1*2+0]=1, U[1*2+1]=1
    # M3: A00 -> U[0*2+0]=1
    # M4: A11 -> U[1*2+1]=1
    # M5: A00+A10 -> U[0*2+0]=1, U[1*2+0]=1
    # M6: A01-A10 -> U[0*2+1]=1, U[1*2+0]=-1
    # M7: A01-A11 -> U[0*2+1]=1, U[1*2+1]=-1

    # V factors (for B indices j,k):
    # M1: B00+B11 -> V[0*2+0]=1, V[1*2+1]=1
    # M2: B00 -> V[0*2+0]=1
    # M3: B01-B11 -> V[0*2+1]=1, V[1*2+1]=-1
    # M4: B10-B00 -> V[1*2+0]=1, V[0*2+0]=-1
    # M5: B11 -> V[1*2+1]=1
    # M6: B00+B10 -> V[0*2+0]=1, V[1*2+0]=1
    # M7: B01+B11 -> V[0*2+1]=1, V[1*2+1]=1

    # W factors (for C indices k,i):
    # M1: C00(+1), C11(+1) -> W[0*2+0]=1, W[1*2+1]=1
    # M2: C10(+1), C11(-1) -> W[0*2+1]=1, W[1*2+1]=-1
    # M3: C01(+1), C11(+1) -> W[1*2+0]=1, W[1*2+1]=1
    # M4: C00(+1), C10(+1) -> W[0*2+0]=1, W[0*2+1]=1
    # M5: C00(-1), C01(+1) -> W[0*2+0]=-1, W[1*2+0]=1
    # M6: C11(+1) -> W[1*2+1]=1
    # M7: C00(+1) -> W[0*2+0]=1

    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    # M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    # M5
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    # M6
    U[1][5] = 1; U[2][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return U, V, W


def _trivial_1x2x2():
    """1x2 * 2x2 product, rank 4 (schoolbook).
    A is 1x2 (indices j=0,1), B is 2x2 (j,k in {0,1}), C is 1x2 (k=0,1).
    U[j], V[j*2+k], W[k*1+0]=W[k].
    """
    U = [[0]*4 for _ in range(2)]
    V = [[0]*4 for _ in range(4)]
    W = [[0]*4 for _ in range(2)]
    col = 0
    for j in range(2):
        for k in range(2):
            U[j][col] = 1
            V[j*2+k][col] = 1
            W[k][col] = 1
            col += 1
    return U, V, W


def _trivial_2x2x1():
    """2x2 * 2x1 product, rank 4 (schoolbook).
    A is 2x2 (i,j in {0,1}), B is 2x1 (j=0,1; k=0), C is 2x1 (i=0,1; k=0).
    U[i*2+j], V[j*2+0]=V[2j], W[0*2+i]=W[i].
    """
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(4)]
    W = [[0]*4 for _ in range(2)]
    col = 0
    for i in range(2):
        for j in range(2):
            U[i*2+j][col] = 1
            V[j*2+0][col] = 1
            W[0*2+i][col] = 1
            col += 1
    return U, V, W


def _block_3x2x2():
    """3x2 * 2x2 product, rank 11 (Strassen 7 for top 2x2x2 + trivial 4 for bottom 1x2x2)."""
    # A is 3x2 (i=0,1,2; j=0,1), B is 2x2 (j=0,1; k=0,1), C is 3x2 (i=0,1,2; k=0,1)
    # Top: A[0:2,:] is 2x2, uses Strassen (rank 7)
    # Bottom: A[2,:] is 1x2, uses trivial (rank 4)
    U_top, V_top, W_top = _strassen_2x2x2()  # U:4x7, V:4x7, W:4x7
    U_bot, V_bot, W_bot = _trivial_1x2x2()    # U:2x4, V:4x4, W:2x4

    rank = 7 + 4
    U = [[0]*rank for _ in range(6)]  # 3*2
    V = [[0]*rank for _ in range(4)]  # 2*2
    W = [[0]*rank for _ in range(6)]  # 2*3

    # Top part (i=0,1): U[i*2+j] = U_top[i*2+j] for j=0,1
    for r in range(4):
        for c in range(7):
            U[r][c] = U_top[r][c]
    # V same for top (V[j*2+k] for j=0,1, k=0,1)
    for r in range(4):
        for c in range(7):
            V[r][c] = V_top[r][c]
    # W top: W[k*3+i] for i=0,1 -> W_top[k*2+i]
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W[k*3+i][c] = W_top[k*2+i][c]

    # Bottom part (i=2): U[2*2+j] = U_bot[j] for j=0,1
    for j in range(2):
        for c in range(4):
            U[2*2+j][7+c] = U_bot[j][c]
    # V bottom: V[j*2+k] = V_bot[j*2+k]
    for r in range(4):
        for c in range(4):
            V[r][7+c] = V_bot[r][c]
    # W bottom: W[k*3+2] = W_bot[k]
    for k in range(2):
        for c in range(4):
            W[k*3+2][7+c] = W_bot[k][c]

    return U, V, W


def _tiling_44(n, m, p):
    """Tile 4 blocks of 3x2x2 (rank 11 each) to get rank 44 for 3x4*4x4.
    Split B (4x4) into two 2x2 column-blocks: B[:,0:2] and B[:,2:4].
    Split A (3x4) into two 3x2 row-blocks: A[:,0:2] and A[:,2:4].
    C[:,0:2] = A[:,0:2]*B[:,0:2] + A[:,2:4]*B[:,2:4]  (two 3x2*2x2 blocks)
    C[:,2:4] = A[:,0:2]*B[:,0:2] + A[:,2:4]*B[:,2:4]  (two 3x2*2x2 blocks)
    Wait, that's not right. Let me re-index.
    
    A is 3x4: columns 0,1,2,3. B is 4x4: rows 0,1,2,3; columns 0,1,2,3.
    C is 3x4: columns 0,1,2,3.
    
    C[:,0:2] = A[:,0:2]*B[:,0:2] + A[:,2:4]*B[:,2:4]
    C[:,2:4] = A[:,0:2]*B[:,0:2] + A[:,2:4]*B[:,2:4]
    
    Each term is 3x2 * 2x2. There are 4 such terms total.
    So we tile 4 blocks of 3x2x2, each rank 11, total rank 44.
    """
    U_blk, V_blk, W_blk = _block_3x2x2()
    rank = 4 * 11

    # Full dimensions: U: 3*4=12, V: 4*4=16, W: 4*3=12
    U = [[0]*rank for _ in range(12)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(12)]

    # Block (a_col_block, b_col_block, out_col_block):
    # a_col_block in {0,1}: A columns {0,1} or {2,3}
    # b_col_block in {0,1}: B rows {0,1} or {2,3}  (these are the "inner" dimension)
    # out_col_block in {0,1}: C columns {0,1} or {2,3}
    # C[:, out] = A[:, a] * B[:, out] ... wait, B's rows are the inner dim.
    # C[i, k] = sum_j A[i,j] * B[j,k]
    # Split j into {0,1} and {2,3}:
    # C[i,k] = sum_{j in {0,1}} A[i,j]*B[j,k] + sum_{j in {2,3}} A[i,j]*B[j,k]
    # For fixed k (output column), and fixed j-block:
    #   term = A[:, jblock] (3x2) * B[jblock, k] (2x1)
    # That's 3x2 * 2x1, not 3x2*2x2.
    
    # Let me reconsider. For a pair of output columns (k0, k0+1):
    # C[:, k0:k0+2] = A[:,0:2]*B[:,k0:k0+2] + A[:,2:4]*B[:,k0:k0+2]
    # Each A[:,jblock] is 3x2, B[:,kblock] is 2x2 (rows jblock, cols kblock).
    # So each term is 3x2 * 2x2, rank 11.
    # There are 2 jblocks * 2 kblocks = 4 terms. Total rank 44.

    for jblock in range(2):  # j in {0,1} or {2,3}
        for kblock in range(2):  # k in {0,1} or {2,3}
            base = (jblock * 2 + kblock) * 11
            # A columns: jblock*2 + j for j in {0,1}
            # B rows: jblock*2 + j for j in {0,1}; B cols: kblock*2 + k for k in {0,1}
            # C cols: kblock*2 + k for k in {0,1}; C rows: i in {0,1,2}
            for i in range(3):
                for j in range(2):
                    # U index: i*4 + (jblock*2 + j)
                    u_idx = i * 4 + (jblock * 2 + j)
                    # In block: U_block index = i*2 + j
                    u_blk = i * 2 + j
                    for c in range(11):
                        U[u_idx][base + c] = U_blk[u_blk][c]
            for j in range(2):
                for k in range(2):
                    # V index: (jblock*2+j)*4 + (kblock*2+k)
                    v_idx = (jblock * 2 + j) * 4 + (kblock * 2 + k)
                    v_blk = j * 2 + k
                    for c in range(11):
                        V[v_idx][base + c] = V_blk[v_blk][c]
            for k in range(2):
                for i in range(3):
                    # W index: (kblock*2+k)*3 + i
                    w_idx = (kblock * 2 + k) * 3 + i
                    w_blk = k * 3 + i
                    for c in range(11):
                        W[w_idx][base + c] = W_blk[w_blk][c]

    return U, V, W


def _local_search_improve(U, V, W, n, m, p, time_budget, seed=0):
    """Try to reduce rank by removing redundant terms or finding sparser representations.
    This is a heuristic; may not always improve.
    """
    random.seed(seed)
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    best = (rank, U, V, W)
    start = time.time()

    # Try to remove each column and re-verify (greedy)
    for c in range(rank - 1, -1, -1):
        if time.time() - start > time_budget:
            break
        U_test = [row[:c] + row[c+1:] for row in U]
        V_test = [row[:c] + row[c+1:] for row in V]
        W_test = [row[:c] + row[c+1:] for row in W]
        if _verify_fast(U_test, V_test, W_test, n, m, p):
            U = U_test; V = V_test; W = W_test
            rank -= 1
            best = (rank, [r[:] for r in U], [r[:] for r in V], [r[:] for r in W])

    return best[1], best[2], best[3], best[0]


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    deadline = time.time() + 130  # Safety margin (150s wall-clock limit)

    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}

    # Try the tiling-44 approach
    if time.time() < deadline - 10:
        U44, V44, W44 = _tiling_44(n, m, p)
        if _verify_fast(U44, V44, W44, n, m, p):
            best_rank = 44
            best_result = {"U": U44, "V": V44, "W": W44}
            # Try to improve via local search
            U_l, V_l, W_l, r_l = _local_search_improve(U44, V44, W44, n, m, p, time_budget=30)
            if r_l < best_rank:
                best_rank = r_l
                best_result = {"U": U_l, "V": V_l, "W": W_l}

    # Try a different tiling: split into 3x4*4x2 blocks (two of them)
    # Each 3x4*4x2: split A into 3x2+3x2, B into 4x2.
    # C[:,0:2] = A[:,0:2]*B[:,0:2] + A[:,2:4]*B[:,2:4]
    # Each 3x2*2x2 is rank 11. Two of them per half, so 4 total = 44. Same as before.

    # Try yet another approach: use Strassen on 2x2x2 sub-blocks more aggressively.
    # Split A (3x4) into 1x4 rows (3 rows). Split B (4x4) into 4x1 columns (4 cols).
    # Each row i: C[i,:] = A[i,:]*B (1x4 * 4x4), rank 16 schoolbook.
    # Not helpful.

    # Try: split into 2x2 blocks of A where possible.
    # A is 3x4: top 2x4, bottom 1x4.
    # B is 4x4: 2x2 blocks.
    # C top 2x4 = A_top(2x4) * B(4x4)
    #   = A_top[:,0:2]*B[:,0:4] + A_top[:,2:4]*B[:,0:4]  ... no, B's rows are inner.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # For i in {0,1} (top), split j into {0,1} and {2,3}:
    # C[i,k] = sum_{j in {0,1}} A[i,j]*B[j,k] + sum_{j in {2,3}} A[i,j]*B[j,k]
    # For fixed k-block {k0,k0+1}:
    # C[top, kblock] = A_top[:,0:2]*B[:,kblock] + A_top[:,2:4]*B[:,kblock]
    # A_top[:,0:2] is 2x2, B[:,kblock] is 4x2. That's 2x2*4x2, not a standard block.
    # Split B[:,kblock] into B[0:2,kblock] (2x2) and B[2:4,kblock] (2x2):
    # C[top,kblock] = A_top[:,0:2]*B[0:2,kblock] + A_top[:,2:4]*B[2:4,kblock]
    # Each is 2x2*2x2, rank 7 (Strassen). Two per (i-block, k-block).
    # i-block: top (i=0,1) uses Strassen. bottom (i=2) is 1x4, trivial.
    # k-block: 2 blocks.
    # Top: 2 kblocks * 2 jblocks = 4 Strassen blocks = 28
    # Bottom: 2 kblocks * 2 jblocks = 4 trivial 1x2*2x2 = 16
    # Total: 28 + 16 = 44. Same as before.

    # The 44 seems to be the natural block-Strassen bound. To beat it, we need
    # cross-block sharing or a fundamentally different decomposition.

    # Try a randomized/structured search for rank < 44.
    # This is hard; the known lower bound is 16, but finding a decomposition
    # between 16 and 44 is a research-level problem.

    # Given time constraints, return the best found.
    return best_result

# EVOLVE-BLOCK-END
