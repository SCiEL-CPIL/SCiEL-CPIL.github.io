"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using numpy broadcasting.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Check T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    U_arr = U.reshape(n, m, r)
    V_arr = V.reshape(m, p, r)
    W_arr = W.reshape(p, n, r)

    Ue = U_arr[:, :, None, :]  # (n, m, 1, r)
    Ve = V_arr[None, :, :, :]  # (1, m, p, r)
    We = W_arr.T[:, None, :, :]  # (n, 1, p, r)

    T = np.sum(Ue * Ve * We, axis=3)  # (n, m, p)
    return np.allclose(T, np.ones((n, m, p)), atol=1e-8)

def _verify_exact(U, V, W, n, m, p):
    """Exact verification with integer arithmetic (entries should be half-integers)."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    U2 = (2 * U).round()
    V2 = (2 * V).round()
    W2 = (2 * W).round()

    if not (np.allclose(U2, 2 * U, atol=1e-6) and
            np.allclose(V2, 2 * V, atol=1e-6) and
            np.allclose(W2, 2 * W, atol=1e-6)):
        return False

    U2 = U2.astype(np.int64)
    V2 = V2.astype(np.int64)
    W2 = W2.astype(np.int64)

    U_arr = U2.reshape(n, m, r)
    V_arr = V2.reshape(m, p, r)
    W_arr = W2.reshape(p, n, r)

    Ue = U_arr[:, :, None, :]
    Ve = V_arr[None, :, :, :]
    We = W_arr.T[:, None, :, :]

    T = np.sum(Ue * Ve * We, axis=3)  # (n, m, p)
    return np.all(T == 8)  # scaled by 2^3

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
    return U, V, W

def _strassen_2x2():
    """Return (U,V,W) factor matrices (each of shape (4,7)) for 2x2*2x2 via Strassen,
    in the convention U[i*2+j], V[j*2+k], W[k*2+i] for a 2x2 block.
    Strassen bilinear forms (indices 0..3 for A, 0..3 for B, 0..3 for C):
    A: A00=0, A01=1, A10=2, A11=3
    B: B00=0, B01=1, B10=2, B11=3
    C: C00=0, C01=1, C10=2, C11=3
    """
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11) B00
    # M3 = A00 (B10 - B11)
    # M4 = A11 (B00 - B01)
    # M5 = (A00+A01) B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # M1
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    # M2
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1; W[3,1]=1
    # M3
    U[0,2]=1
    V[2,2]=1; V[3,2]=-1
    W[1,2]=1; W[3,2]=1
    # M4
    U[3,3]=1
    V[0,3]=1; V[1,3]=-1
    W[0,3]=1; W[2,3]=1
    # M5
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1; W[1,4]=1
    # M6
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[2,5]=1
    # M7
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1; W[1,6]=-1
    return U, V, W

def _embed_2x2_into_3x4_4x4(U2, V2, W2, i_off, j_off, k_off, n, m, p):
    """Embed a 2x2 block factor (for A[i_off:i_off+2, j_off:j_off+2] * B[j_off:j_off+2, k_off:k_off+2])
    into the full 3x4*4x4 factor matrices (as a contribution to a set of rank columns).
    Returns (U_full, V_full, W_full) of shapes (12,7),(16,7),(12,7).
    """
    Uf = np.zeros((n * m, 7))
    Vf = np.zeros((m * p, 7))
    Wf = np.zeros((p * n, 7))
    for a in range(4):
        ii = a // 2
        jj = a % 2
        Uf[(i_off + ii) * m + (j_off + jj), :] = U2[a, :]
    for b in range(4):
        jj = b // 2
        kk = b % 2
        Vf[(j_off + jj) * p + (k_off + kk), :] = V2[b, :]
    for c in range(4):
        kk = c // 2
        ii = c % 2
        Wf[(k_off + kk) * n + (i_off + ii), :] = W2[c, :]
    return Uf, Vf, Wf

def _try_block_strassen(n, m, p, deadline):
    """
    Structured 'transposed Strassen' block decomposition for 3x4 * 4x4.
    Partition A (3x4) into rows [0:2, 2:3] and cols [0:2, 2:4];
    Partition B (4x4) into rows [0:2, 2:4] and cols [0:2, 2:4].
    Use Strassen (rank 7) for the four 2x2*2x2 blocks, then a linear solve
    to combine/overlap columns and reduce total rank.
    """
    U2, V2, W2 = _strassen_2x2()

    # Four 2x2 blocks of the 4x4 middle/output space.
    # Block (I,J) where I in {0:2, 2:3} for A rows, J in {0:2,2:4} for A cols / B rows, K in {0:2,2:4} for B cols.
    # Actually A is 3x4, B is 4x4. Split A rows into [0,1] and [2]; A cols into [0,1] and [2,3].
    # B rows into [0,1] and [2,3]; B cols into [0,1] and [2,3].
    # The 2x2 Strassen applies to A[0:2,0:2]*B[0:2,0:2], A[0:2,2:4]*B[2:4,0:2], etc.
    # But A row [2] is only 1 row, not 2. So we can't directly use 2x2 Strassen for the bottom A row.

    # Better: split A into top 2 rows and bottom 1 row.
    # Top part: A[0:2, :] (2x4) * B[:, :] (4x4) -> 2x4 output.
    #   Split cols into [0:2] and [2:4]:
    #     A[0:2,0:2] (2x2) * B[0:2,:] (2x4) -> use Strassen on the 2x2 * 2x2 left half of B, and 2x2*2x2 right half.
    #     A[0:2,2:4] (2x2) * B[2:4,:] (2x4) -> same.
    # Bottom part: A[2:3, :] (1x4) * B[:, :] (4x4) -> 1x4 output.
    #   This is just 4 scalar inner products of length 4 -> rank 4 (schoolbook) or use a 1x4*4x4 trick.

    # Let's just construct the top 2 rows with Strassen blocks and bottom row with schoolbook,
    # then attempt a global linear solve to reduce rank.

    # Top 2 rows, left 2 cols (A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2])
    U_tl, V_tl, W_tl = _embed_2x2_into_3x4_4x4(U2, V2, W2, 0, 0, 0, n, m, p)
    # Top 2 rows, right 2 cols (A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2])  [B rows 2:4, cols 0:2]
    U_tr, V_tr, W_tr = _embed_2x2_into_3x4_4x4(U2, V2, W2, 0, 2, 0, n, m, p)
    # Top 2 rows, left 2 cols of B but output right 2 cols (A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4])
    U_tl2, V_tl2, W_tl2 = _embed_2x2_into_3x4_4x4(U2, V2, W2, 0, 0, 2, n, m, p)
    # Top 2 rows, right 2 cols (A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4])
    U_tr2, V_tr2, W_tr2 = _embed_2x2_into_3x4_4x4(U2, V2, W2, 0, 2, 2, n, m, p)

    # Concatenate these 4 blocks (each rank 7) -> rank 28 for top 2 rows.
    U_top = np.hstack([U_tl, U_tr, U_tl2, U_tr2])
    V_top = np.hstack([V_tl, V_tr, V_tl2, V_tr2])
    W_top = np.hstack([W_tl, W_tr, W_tl2, W_tr2])

    # Bottom 1 row (A[2:3, :]) schoolbook: 4 terms (j=0..3), each contributing to all k.
    U_bot = np.zeros((n * m, 4))
    V_bot = np.zeros((m * p, 4))
    W_bot = np.zeros((p * n, 4))
    for j in range(m):
        for k in range(p):
            c = j * p + k
            U_bot[2 * m + j, c] = 1.0
            V_bot[j * p + k, c] = 1.0
            W_bot[k * n + 2, c] = 1.0

    U_full = np.hstack([U_top, U_bot])
    V_full = np.hstack([V_top, V_bot])
    W_full = np.hstack([W_top, W_bot])

    if not _verify_fast(U_full, V_full, W_full, n, m, p):
        return None
    if not _verify_exact(U_full, V_full, W_full, n, m, p):
        return None

    r = U_full.shape[1]
    return U_full, V_full, W_full, r

def _try_numeric_refinement(n, m, p, U0, V0, W0, deadline):
    """Try to reduce rank via column-merging / numerical optimization starting from U0,V0,W0."""
    start = time.time()
    U = U0.copy()
    V = V0.copy()
    W = W0.copy()
    r = U.shape[1]

    # Try to zero out small entries and drop zero columns
    def _drop_zero_cols(U, V, W, tol=1e-9):
        mask = np.any(np.abs(U) > tol, axis=0) | np.any(np.abs(V) > tol, axis=0) | np.any(np.abs(W) > tol, axis=0)
        return U[:, mask], V[:, mask], W[:, mask]

    U, V, W = _drop_zero_cols(U, V, W)
    if _verify_exact(U, V, W, n, m, p):
        return U, V, W, U.shape[1]
    return None

def _try_linear_solve_reduction(n, m, p, U, V, W, deadline):
    """
    Given a valid decomposition of rank r, try to find a lower-rank one by
    expressing some columns as linear combinations of others (if the columns
    are linearly dependent in a way that preserves the tensor identity).
    This is heuristic; we attempt to find a subset of columns that still spans.
    """
    start = time.time()
    r = U.shape[1]
    # Try ranks from max lower bound to r-1
    min_rank = max(n * p, m * p, n * m)
    for target in range(min_rank, r):
        if time.time() > deadline - 5:
            break
        # Try to find target columns among the r that form a valid decomposition
        # This is combinatorial; skip for now (too expensive).
        pass
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # Baseline: schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W

    # Try structured block Strassen
    try:
        result = _try_block_strassen(n, m, p, deadline)
        if result is not None:
            Us, Vs, Ws, rs = result
            if rs < best_rank:
                best_rank = rs
                best_U, best_V, best_W = Us, Vs, Ws
    except Exception:
        pass

    # Try numeric refinement / column dropping on the best found
    try:
        result = _try_numeric_refinement(n, m, p, best_U, best_V, best_W, deadline)
        if result is not None:
            Ur, Vr, Wr, rr = result
            if rr < best_rank:
                best_rank = rr
                best_U, best_V, best_W = Ur, Vr, Wr
    except Exception:
        pass

    # Final verification
    if not _verify_fast(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook
        best_U, best_V, best_W = _schoolbook(n, m, p)
        best_rank = n * m * p

    def to_list(mat):
        return np.asarray(mat, dtype=np.float64).tolist()

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}

# EVOLVE-BLOCK-END
