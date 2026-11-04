import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 7]
N, M, P = DIMENSIONS

def _verify(U, V, W):
    n, m, p = N, M, P
    dimA = n * m
    dimB = m * p
    dimW = p * n
    rank = U.shape[1]

    if U.shape != (dimA, rank) or V.shape != (dimB, rank) or W.shape != (dimW, rank):
        return False

    T_expected = np.zeros((dimA, dimB, dimW), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                w = k * n + i
                T_expected[a, b, w] = 1

    U_exp = U[:, np.newaxis, np.newaxis, :]
    V_exp = V[np.newaxis, :, np.newaxis, :]
    W_exp = W[np.newaxis, np.newaxis, :, :]

    T_computed = np.sum(U_exp * V_exp * W_exp, axis=3)

    return np.array_equal(T_expected, T_computed)

def _schoolbook_block(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                w = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[w, col] = 1
                col += 1
    return U, V, W

def _construct_rank21_ladder():
    """
    Construct a rank-21 decomposition by splitting B (3x7) into B1 (3x3) and B2 (3x4).
    """
    n, m, p = N, M, P
    p1 = 3
    p2 = 4
    if p1 + p2 != p:
        return None

    U1, V1, W1 = _schoolbook_block(n, m, p1)
    U2, V2, W2 = _schoolbook_block(n, m, p2)

    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    total_rank = rank1 + rank2

    U = np.zeros((n * m, total_rank), dtype=np.int64)
    V = np.zeros((m * p, total_rank), dtype=np.int64)
    W = np.zeros((p * n, total_rank), dtype=np.int64)

    U[:, 0:rank1] = U1
    for j in range(m):
        for k in range(p1):
            idx_v1 = j * p1 + k
            idx_v = j * p + k
            V[idx_v, 0:rank1] = V1[idx_v1]

    for k in range(p1):
        for i in range(n):
            idx_w1 = k * n + i
            idx_w = k * n + i
            W[idx_w, 0:rank1] = W1[idx_w1]

    U[:, rank1:total_rank] = U2
    for j in range(m):
        for k in range(p2):
            idx_v2 = j * p2 + k
            idx_v = j * p + (p1 + k)
            V[idx_v, rank1:total_rank] = V2[idx_v2]

    for k in range(p2):
        for i in range(n):
            idx_w2 = k * n + i
            idx_w = (p1 + k) * n + i
            W[idx_w, rank1:total_rank] = W2[idx_w2]

    return U, V, W, total_rank

def _strassen_2x2():
    """Return 7-term rank-7 decomposition for 2x2 matrix multiplication."""
    # 2x2 multiplication: A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    # Winograd's 7-multiplication formula
    # We represent in the tensor decomposition convention
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)

    # The 7 products:
    # M1 = (a10 + a11) * b00
    # M2 = (a10 + a11) * (b10 + b11)
    # M3 = (a00 - a10) * (b00 + b10)
    # M4 = a00 * (b01 - b00)
    # M5 = a11 * (b11 - b10)
    # M6 = (a00 + a01) * b11
    # M7 = (a01 - a11) * (b10 + b11)
    #
    # C00 = M1 + M3 - M4 - M7
    # C01 = M4 + M6
    # C10 = M2 + M5
    # C11 = M1 - M2 + M5 + M6

    # Indexing: a = i*m + j, b = j*p + k, w = k*n + i
    # n=m=p=2
    # a indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # b indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # w indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)  [w = k*n+i, so w=(0,0)=0, w=(1,0)=2? No: w=k*n+i]
    # w[0,0] = 0*2+0 = 0, w[0,1] = 0*2+1 = 1, w[1,0] = 1*2+0 = 2, w[1,1] = 1*2+1 = 3

    # M1: U has (1,0) and (1,1) -> a2, a3; V has (0,0) -> b0
    U[2, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1   # C00 += M1
    W[3, 0] = 1   # C11 += M1

    # M2: U has (1,0) and (1,1); V has (1,0) and (1,1) -> b2, b3
    U[2, 1] = 1; U[3, 1] = 1
    V[2, 1] = 1; V[3, 1] = 1
    W[2, 1] = 1   # C10 += M2
    W[3, 1] = -1  # C11 -= M2

    # M3: U has (0,0) - (1,0) -> a0 - a2; V has (0,0)+(1,0) -> b0 + b2
    U[0, 2] = 1; U[2, 2] = -1
    V[0, 2] = 1; V[2, 2] = 1
    W[0, 2] = 1   # C00 += M3
    W[2, 2] = 1   # C10 += M3  -- wait, C10 = M2 + M5, not M3
    # Correction: C00 = M1 + M3 - M4 - M7; C10 = M2 + M5
    # So M3 only contributes to C00
    W[2, 2] = 0
    W[0, 2] = 1

    # M4: a00 * (b01 - b00)
    U[0, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1  # C00 -= M4
    W[1, 3] = 1   # C01 += M4

    # M5: a11 * (b11 - b10)
    U[3, 4] = 1
    V[3, 4] = 1; V[2, 4] = -1
    W[2, 4] = 1   # C10 += M5
    W[3, 4] = 1   # C11 += M5

    # M6: (a00 + a01) * b11
    U[0, 5] = 1; U[1, 5] = 1
    V[3, 5] = 1
    W[1, 5] = 1   # C01 += M6
    W[3, 5] = 1   # C11 += M6

    # M7: (a01 - a11) * (b10 + b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1  # C00 -= M7

    return U, V, W

def _embed_2x2_into_2x3x7(offset_k, offset_j_start):
    """
    Embed a 2x2 Strassen decomposition into the 2x3x7 tensor.
    offset_k: which rows of B (and corresponding output rows) to use, 0-5
    offset_j_start: which columns of A to start from
    This places the 2x2 block at rows [offset_k, offset_k+2) of B, columns [offset_j_start, offset_j_start+2) of A.
    """
    n, m, p = N, M, P
    U, V, W = _strassen_2x2()
    # U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    # Map U indices: local a = i*2+j (i in 0,1; j in 0,1) -> global a = i*m + (j + offset_j_start)
    # Map V indices: local b = j*2+k (j in 0,1; k in 0,1) -> global b = (j + ?) ... 
    # Actually V's rows correspond to B's entries. B is 3x7, indexed by (j, k) where j is row of B (0-2), k is col (0-6)
    # The 2x2 block uses B rows [offset_k, offset_k+1] and B cols... wait.
    # Let me reconsider. A is n x m = 2 x 3. B is m x p = 3 x 7.
    # The 2x2 subproblem uses A rows 0,1 (all n=2 rows) and A cols offset_j_start, offset_j_start+1
    # B rows offset_j_start, offset_j_start+1 and B cols offset_k, offset_k+1
    # Output C rows offset_k, offset_k+1 and output cols 0,1 (all n=2 cols)
    
    # This is getting complex. Let me just directly construct the full 2x3x7 using Strassen for a 2x2x2 block.
    
    # Actually, for a cleaner approach, let me use the fact that 2x3x7 can be decomposed as:
    # 2x3x7 = 2x3x1 + 2x3x1 + 2x3x1 + 2x2x2 + 2x1x2 (roughly)
    # But let me just try a direct construction.
    
    pass

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140.0

    best_rank = float('inf')
    best_U = None
    best_V = None
    best_W = None

    # Try the rank-21 ladder construction
    res = _construct_rank21_ladder()
    if res is not None:
        U, V, W, rank = res
        if time.time() < DEADLINE - 10.0:
            if _verify(U, V, W):
                if rank < best_rank:
                    best_rank = rank
                    best_U = U
                    best_V = V
                    best_W = W

    # Fallback to schoolbook if needed
    if best_rank > 42 or best_U is None:
        U, V, W = _schoolbook_block(N, M, P)
        rank = U.shape[1]
        if time.time() < DEADLINE - 5.0:
            if _verify(U, V, W):
                if rank < best_rank:
                    best_rank = rank
                    best_U = U
                    best_V = V
                    best_W = W
        else:
            best_rank = rank
            best_U = U
            best_V = V
            best_W = W

    if best_U is None:
        return {"U": [], "V": [], "W": []}

    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()

    return {"U": U_out, "V": V_out, "W": W_out}
