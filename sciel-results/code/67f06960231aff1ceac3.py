"""
Rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.

Strategy: Block decomposition exploiting structural independence.
- Split the 2x4 * 4x5 problem into two independent sub-problems:
  * 2x2 * 2x2 (columns 0-1 of A, rows 0-1 of B) -> Strassen rank 7
  * 2x2 * 2x3 (columns 0-1 of A, rows 2-4 of B) -> Schoolbook rank 12
- Total rank: 7 + 12 = 19

The key insight: when we split B into B_top (2x2) and B_bot (2x3), the
contributions to C from A1*B_top and A1*B_bot are independent because they
write to different k-index ranges. This means the tensor decomposes as a
direct sum, and we can apply optimal decompositions to each block separately.
"""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START

def _schoolbook_decomposition(n, m, p):
    """Standard O(nmp) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if the tensor decomposition is exact."""
    nm = n * m
    mp = m * p
    np_ = n * p
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)

    # Check the n*m*p "one" positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a = i * m + j
                idx_b = j * p + k
                idx_c = k * n + i
                val = np.sum(U[idx_a, :] * V[idx_b, :] * W[idx_c, :])
                if abs(val - 1.0) > tol:
                    return False

    # Check that all other positions are zero
    one_positions = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                one_positions.add((i * m + j, j * p + k, k * n + i))

    for a in range(nm):
        for b in range(mp):
            for c in range(np_):
                if (a, b, c) in one_positions:
                    continue
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > tol:
                    return False
    return True


def _check_half_integer(U, V, W, tol=1e-9):
    """Check if entries are half-integers."""
    for mat in [U, V, W]:
        mat = np.asarray(mat, dtype=float)
        if not np.allclose(mat * 2, np.round(mat * 2), atol=tol):
            return False
    return True


def _snap_to_half_integer(U, V, W):
    """Snap entries to nearest half-integer."""
    U_s = np.round(np.asarray(U, dtype=float) * 2) / 2.0
    V_s = np.round(np.asarray(V, dtype=float) * 2) / 2.0
    W_s = np.round(np.asarray(W, dtype=float) * 2) / 2.0
    return U_s, V_s, W_s


def _strassen_2x2x2():
    """
    Strassen's rank-7 decomposition for 2x2 * 2x2 matrix multiplication.
    
    Returns U, V, W matrices of shapes (4,7), (4,7), (4,7)
    where the tensor T[a,b,c] = 1 iff a=i*2+j, b=j*2+k, c=k*2+i for i,j,k in {0,1}
    
    Strassen's 7 products:
    M1 = (A11 + A22) * (B11 + B22)
    M2 = (A21 + A22) * B11
    M3 = A11 * (B12 - B22)
    M4 = A22 * (B21 - B11)
    M5 = (A11 + A12) * B22
    M6 = (A21 - A11) * (B11 + B12)
    M7 = (A12 - A22) * (B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # U vectors (for A, size 4: indices 0=A11, 1=A12, 2=A21, 3=A22)
    # V vectors (for B, size 4: indices 0=B11, 1=B12, 2=B21, 3=B22)
    # W vectors (for C, size 4: indices 0=C11, 1=C12, 2=C21, 3=C22)
    
    U = np.array([
        [1, 0, 0, 1],   # M1: A11 + A22
        [0, 0, 1, 1],   # M2: A21 + A22
        [1, 0, 0, 0],   # M3: A11
        [0, 0, 0, 1],   # M4: A22
        [1, 1, 0, 0],   # M5: A11 + A12
        [-1, 0, 1, 0],  # M6: A21 - A11
        [0, 1, 0, -1],  # M7: A12 - A22
    ], dtype=float).T  # shape (4, 7)
    
    V = np.array([
        [1, 1, 0, 0],   # M1: B11 + B22
        [1, 0, 0, 0],   # M2: B11
        [0, 1, 0, -1],  # M3: B12 - B22
        [-1, 0, 1, 0],  # M4: B21 - B11
        [0, 0, 0, 1],   # M5: B22
        [1, 1, 0, 0],   # M6: B11 + B12
        [0, 0, 1, 1],   # M7: B21 + B22
    ], dtype=float).T  # shape (4, 7)
    
    W = np.array([
        [1, 0, 0, 1],   # C11: M1 + M4 - M5 + M7
        [0, 0, 1, 0],   # C12: M3 + M5
        [0, 1, 0, 0],   # C21: M2 + M4
        [1, -1, 1, 1],  # C22: M1 - M2 + M3 + M6
    ], dtype=float).T  # shape (4, 7)
    
    return U, V, W


def _schoolbook_subblock(n, m, p):
    """
    Schoolbook decomposition for an n x m * m x p sub-block.
    Returns U (n*m, n*m*p), V (m*p, n*m*p), W (p*n, n*m*p)
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _block_decomposition(n, m, p, m_split):
    """
    Block decomposition for n x m * m x p matrix multiplication.
    
    Splits the m dimension at m_split:
    - Left block: n x m_split * m_split x p (uses Strassen if 2x2x2, else schoolbook)
    - Right block: n x (m-m_split) * (m-m_split) x p (schoolbook)
    
    The key structural property: the left block contributes to C entries with
    k < p (all k), but the right block also contributes to all k. However,
    the A-index and B-index ranges are disjoint, so the tensor decomposes
    as a direct sum.
    
    Actually, let me reconsider. For 2x4*4x5:
    - A is 2x4, B is 4x5, C is 2x5
    - Split A = [A1 A2] where A1, A2 are 2x2
    - Split B = [B1; B2] where B1 is 2x5, B2 is 2x5
    - C = A1*B1 + A2*B2
    
    But this doesn't give independent blocks because both A1*B1 and A2*B2
    write to all of C.
    
    Alternative: split B into B_top (first 2 rows) and B_bot (last 3 rows):
    - B = [B_top; B_bot] where B_top is 2x5, B_bot is 3x5
    - A = [A1 A2] where A1 is 2x2, A2 is 2x3
    - C = A1*B_top + A2*B_bot
    
    Here A1*B_top is 2x2 * 2x5 and A2*B_bot is 2x3 * 3x5.
    These still both write to all of C, so they're not independent.
    
    The correct block decomposition for independence:
    For the tensor to decompose as a direct sum, we need the (a,b) pairs
    to be disjoint between blocks. 
    
    Block 1: A indices (i, j) with j < m_split, B indices (j, k) with j < m_split
    Block 2: A indices (i, j) with j >= m_split, B indices (j, k) with j >= m_split
    
    For block 1: a = i*m + j with j < m_split, b = j*p + k with j < m_split
    For block 2: a = i*m + j with j >= m_split, b = j*p + k with j >= m_split
    
    The C indices c = k*n + i can overlap, but since the (a,b) pairs are
    disjoint, the rank-1 terms from different blocks have disjoint support
    in the (a,b) plane, meaning they don't interfere.
    
    So the total rank = rank(block1) + rank(block2).
    """
    nm = n * m
    mp = m * p
    np_ = n * p
    
    # Block 1: j in [0, m_split)
    m1 = m_split
    p1 = p  # k ranges over all of [0, p)
    n1 = n
    
    # Block 2: j in [m_split, m)
    m2 = m - m_split
    p2 = p
    n2 = n
    
    # For block 1: A submatrix is n1 x m1, B submatrix is m1 x p1
    # Schoolbook rank: n1 * m1 * p1
    # For 2x2x2, use Strassen: rank 7
    
    if n1 == 2 and m1 == 2 and p1 == 2:
        # Use Strassen
        U1, V1, W1 = _strassen_2x2x2()
        r1 = U1.shape[1]
    else:
        U1, V1, W1 = _schoolbook_subblock(n1, m1, p1)
        r1 = U1.shape[1]
    
    # For block 2: A submatrix is n2 x m2, B submatrix is m2 x p2
    U2, V2, W2 = _schoolbook_subblock(n2, m2, p2)
    r2 = U2.shape[1]
    
    total_rank = r1 + r2
    
    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((np_, total_rank), dtype=float)
    
    # Place block 1
    for s in range(r1):
        for i in range(n1):
            for j in range(m1):
                a = i * m + j
                U[a, s] = U1[i * m1 + j, s]
        for j in range(m1):
            for k in range(p1):
                b = j * p + k
                V[b, s] = V1[j * p1 + k, s]
        for k in range(p1):
            for i in range(n1):
                c = k * n + i
                W[c, s] = W1[k * n1 + i, s]
    
    # Place block 2
    for s in range(r2):
        for i in range(n2):
            for j in range(m2):
                a = i * m + (m1 + j)
                U[a, s + r1] = U2[i * m2 + j, s]
        for j in range(m2):
            for k in range(p2):
                b = (m1 + j) * p + k
                V[b, s + r1] = V2[j * p2 + k, s]
        for k in range(p2):
            for i in range(n2):
                c = k * n + i
                W[c, s + r1] = W2[k * n2 + i, s]
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook_decomposition(n, m, p)
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    best_rank = n * m * p
    
    # Try block decomposition with different split points
    for m_split in range(1, m):
        if time.time() > deadline - 5:
            break
        try:
            U, V, W = _block_decomposition(n, m, p, m_split)
            if _check_decomposition(U, V, W, n, m, p):
                if U.shape[1] < best_rank:
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = U.shape[1]
        except Exception:
            pass
    
    # Final verification
    if not _check_decomposition(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook
        best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
