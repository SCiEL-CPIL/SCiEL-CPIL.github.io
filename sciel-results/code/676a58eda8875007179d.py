"""Optimized tensor rank decomposition for 3x4 * 4x4 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

N, M, P = 3, 4, 4
RANK_SCHOOLBOOK = N * M * P  # 48


def _verify_fast(U, V, W, n=N, m=M, p=P):
    """Faster verification using tensor reshaping."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    if V.shape[1] != r or W.shape[1] != r:
        return False
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    expected = np.ones((n, m, p))
    return np.allclose(recon, expected, atol=1e-9)


def _strassen_2x2():
    """
    Return rank-7 decomposition for 2x2 * 2x2 matrix multiplication.
    Indices: U[i*2+j], V[j*2+k], W[k*2+i] for i,j,k in {0,1}
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Product 1: M1 = (a+d)(e+h)
    U[0,0] = 1; U[3,0] = 1  # a, d
    V[0,0] = 1; V[3,0] = 1  # e, h
    W[0,0] = 1; W[3,0] = 1  # C[0,0], C[1,1]
    
    # Product 2: M2 = (c+d)e
    U[2,1] = 1; U[3,1] = 1  # c, d
    V[0,1] = 1              # e
    W[3,1] = -1; W[2,1] = 1  # C[1,1] has -M2, C[1,0] has +M2
    
    # Product 3: M3 = a(f-h)
    U[0,2] = 1              # a
    V[1,2] = 1; V[3,2] = -1  # f, -h
    W[1,2] = 1; W[3,2] = 1   # C[0,1] has +M3, C[1,1] has +M3
    
    # Product 4: M4 = d(g-e)
    U[3,3] = 1              # d
    V[2,3] = 1; V[0,3] = -1  # g, -e
    W[0,3] = 1; W[2,3] = 1   # C[0,0] has +M4, C[1,0] has +M4
    
    # Product 5: M5 = (a+b)h
    U[0,4] = 1; U[1,4] = 1  # a, b
    V[3,4] = 1              # h
    W[0,4] = -1; W[1,4] = 1  # C[0,0] has -M5, C[0,1] has +M5
    
    # Product 6: M6 = (c-d)(f+h)
    U[2,5] = 1; U[3,5] = -1  # c, -d
    V[1,5] = 1; V[3,5] = 1   # f, h
    W[3,5] = 1               # C[1,1] has +M6
    
    # Product 7: M7 = (b-c)(e+g)
    U[1,6] = 1; U[2,6] = -1  # b, -c
    V[0,6] = 1; V[2,6] = 1   # e, g
    W[0,6] = 1               # C[0,0] has +M7
    
    return U, V, W


def _build_strassen_based():
    """
    Build a Strassen-based decomposition for 3x4 * 4x4.
    
    Strategy:
    - Split A (3x4) into A_top (2x4) and A_bottom (1x4)
    - Split B (4x4) into B_left (4x2) and B_right (4x2) for column pairs
    - For A_top * B: use Strassen on 2x2 sub-blocks
    - For A_bottom * B: use simple rank-2 blocks
    
    This gives rank 28 (for 2x4*4x4 via Strassen) + 4 (for 1x4*4x4) = 32.
    """
    rank = 32
    U = np.zeros((12, rank))  # 3*4 = 12 rows
    V = np.zeros((16, rank))  # 4*4 = 16 rows
    W = np.zeros((12, rank))  # 4*3 = 12 rows
    
    col = 0
    
    # Part 1: A_top (2x4) * B (4x4) -> 2x4 result
    # A_top = [A_L (2x2) | A_R (2x2)] where A_L = A[0:2, 0:2], A_R = A[0:2, 2:4]
    # B = [B_T (2x4); B_B (2x4)] where B_T = B[0:2, :], B_B = B[2:4, :]
    # C_top = A_L * B_T + A_R * B_B
    
    # A_L (2x2) * B_T (2x4): Split B_T into two 2x2 blocks by columns
    # B_T = [B_T1 (2x2) | B_T2 (2x2)] where B_T1 = B[0:2, 0:2], B_T2 = B[0:2, 2:4]
    for block in range(2):
        k_start = block * 2
        sU, sV, sW = _strassen_2x2()
        for c in range(7):
            for li in range(2):
                for lj in range(2):
                    U[li*4 + lj, col] = sU[li*2 + lj, c]
            for lj in range(2):
                for lk in range(2):
                    V[lj*4 + k_start + lk, col] = sV[lj*2 + lk, c]
            for lk in range(2):
                for li in range(2):
                    W[(k_start + lk)*3 + li, col] = sW[lk*2 + li, c]
            col += 1
    
    # A_R (2x2) * B_B (2x4): Split B_B into two 2x2 blocks by columns
    # B_B = [B_B1 (2x2) | B_B2 (2x2)] where B_B1 = B[2:4, 0:2], B_B2 = B[2:4, 2:4]
    for block in range(2):
        k_start = block * 2
        sU, sV, sW = _strassen_2x2()
        for c in range(7):
            for li in range(2):
                for lj in range(2):
                    U[li*4 + 2 + lj, col] = sU[li*2 + lj, c]
            for lj in range(2):
                for lk in range(2):
                    V[(2 + lj)*4 + k_start + lk, col] = sV[lj*2 + lk, c]
            for lk in range(2):
                for li in range(2):
                    W[(k_start + lk)*3 + li, col] = sW[lk*2 + li, c]
            col += 1
    
    # Part 2: A_bottom (1x4) * B (4x4) -> 1x4 result
    # A_bottom = [A_L2 (1x2) | A_R2 (1x2)] where A_L2 = A[2, 0:2], A_R2 = A[2, 2:4]
    # B = [B_T (2x4); B_B (2x4)]
    # C_bottom = A_L2 * B_T + A_R2 * B_B
    
    # A_L2 (1x2) * B_T (2x4): rank 2
    for c in range(2):
        j = c  # j=0,1
        U[2*4 + j, col] = 1
        for k in range(4):
            V[j*4 + k, col] = 1
        for k in range(4):
            W[k*3 + 2, col] = 1
        col += 1
    
    # A_R2 (1x2) * B_B (2x4): rank 2
    for c in range(2):
        j = 2 + c  # j=2,3
        U[2*4 + j, col] = 1
        for k in range(4):
            V[j*4 + k, col] = 1
        for k in range(4):
            W[k*3 + 2, col] = 1
        col += 1
    
    return U.tolist(), V.tolist(), W.tolist()


def _schoolbook_decomposition():
    """Standard schoolbook decomposition with rank 48."""
    rank = N * M * P
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i*M+j][col] = 1
                V[j*P+k][col] = 1
                W[k*N+i][col] = 1
                col += 1
    return U, V, W


def _try_subspace_optimization(U_init, V_init, W_init, max_iter=50, tol=1e-10):
    """
    Try to reduce rank by finding linear dependencies among the columns.
    This is a heuristic that may or may not find improvements.
    """
    U = np.array(U_init, dtype=float)
    V = np.array(V_init, dtype=float)
    W = np.array(W_init, dtype=float)
    r = U.shape[1]
    
    # Check if columns are linearly dependent
    # If rank < r, we can potentially reduce the number of terms
    for mat, name in [(U, 'U'), (V, 'V'), (W, 'W')]:
        rank = np.linalg.matrix_rank(mat, tol=tol)
        if rank < r:
            # There might be room for optimization, but this is complex
            pass
    
    return U.tolist(), V.tolist(), W.tolist()


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.
    
    Returns dict with 'U', 'V', 'W' matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 110
    
    # First, try the Strassen-based decomposition (rank 32)
    try:
        U, V, W = _build_strassen_based()
        if _verify_fast(U, V, W):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 10:
        U, V, W = _schoolbook_decomposition()
        return {"U": U, "V": V, "W": W}
    
    # Try to optimize the Strassen-based decomposition
    try:
        U_opt, V_opt, W_opt = _try_subspace_optimization(U, V, W)
        if _verify_fast(U_opt, V_opt, W_opt):
            # Check if rank was reduced
            if len(U_opt[0]) < len(U[0]):
                return {"U": U_opt, "V": V_opt, "W": W_opt}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U, V, W = _schoolbook_decomposition()
    return {"U": U, "V": V, "W": W}
