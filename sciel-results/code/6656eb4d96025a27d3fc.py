"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s wall-clock

    # Build Strassen-based decomposition
    result = _build_strassen_3x5x6(n, m, p)
    
    if result is not None:
        U, V, W, rank = result
        # Verify
        if _verify_fast(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}


def _get_strassen_2x2x2():
    """
    Return U, V, W factors for 2x2x2 matrix multiplication (Strassen, rank 7).
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # U: A factors
    # M0: A00, A11
    U[0, 0] = 1
    U[3, 0] = 1
    # M1: A10, A11
    U[2, 1] = 1
    U[3, 1] = 1
    # M2: A00
    U[0, 2] = 1
    # M3: A11
    U[3, 3] = 1
    # M4: A00, A01
    U[0, 4] = 1
    U[1, 4] = 1
    # M5: A10, -A00
    U[2, 5] = 1
    U[0, 5] = -1
    # M6: A01, -A11
    U[1, 6] = 1
    U[3, 6] = -1
    
    # V: B factors
    # M0: B00, B11
    V[0, 0] = 1
    V[3, 0] = 1
    # M1: B00
    V[0, 1] = 1
    # M2: B01, -B11
    V[1, 2] = 1
    V[3, 2] = -1
    # M3: B10, -B00
    V[2, 3] = 1
    V[0, 3] = -1
    # M4: B11
    V[3, 4] = 1
    # M5: B00, B01
    V[0, 5] = 1
    V[1, 5] = 1
    # M6: B10, B11
    V[2, 6] = 1
    V[3, 6] = 1
    
    # W: C factors (W[k*2+i, c])
    # C00 = M0 + M3 - M4 + M6
    W[0, 0] = 1
    W[0, 3] = 1
    W[0, 4] = -1
    W[0, 6] = 1
    
    # C01 = M2 + M4
    # Index 1 is k=0, i=1
    W[1, 2] = 1
    W[1, 4] = 1
    
    # C10 = M1 + M3
    # Index 2 is k=1, i=0
    W[2, 1] = 1
    W[2, 3] = 1
    
    # C11 = M0 - M1 + M2 + M5
    # Index 3 is k=1, i=1
    W[3, 0] = 1
    W[3, 1] = -1
    W[3, 2] = 1
    W[3, 5] = 1
    
    return U, V, W


def _build_strassen_3x5x6(n, m, p):
    """
    Build tensor decomposition for 3x5 @ 5x6 by projecting a 6x6x6
    Strassen decomposition (rank 63) onto the 3x5x6 subspace.
    """
    sU, sV, sW = _get_strassen_2x2x2()
    
    # Build full 6x6x6 Strassen decomposition: rank = 3*3*7 = 63
    full_rank = 63
    full_U = np.zeros((36, full_rank))  # 6*6 = 36
    full_V = np.zeros((36, full_rank))  # 6*6 = 36
    full_W = np.zeros((36, full_rank))  # 6*6 = 36
    
    col = 0
    for i_block in range(3):
        for j_block in range(3):
            for k_block in range(3):
                for c in range(7):
                    # U: A[i_block*2 + di, j_block*2 + dj]
                    for di in range(2):
                        for dj in range(2):
                            full_U[(i_block*2 + di)*6 + (j_block*2 + dj), col] = sU[di*2 + dj, c]
                    # V: B[j_block*2 + dj, k_block*2 + dk]
                    for dj in range(2):
                        for dk in range(2):
                            full_V[(j_block*2 + dj)*6 + (k_block*2 + dk), col] = sV[dj*2 + dk, c]
                    # W: C[k_block*2 + dk, i_block*2 + di]
                    for dk in range(2):
                        for di in range(2):
                            full_W[(k_block*2 + dk)*6 + (i_block*2 + di), col] = sW[dk*2 + di, c]
                    col += 1
    
    # Project to 3x5x6:
    # U: rows (i,j) for i in 0..2, j in 0..4 -> 15 rows
    # V: rows (j,k) for j in 0..4, k in 0..5 -> 30 rows
    # W: rows (k,i) for k in 0..5, i in 0..2 -> 18 rows
    
    U_rows = []
    for i in range(n):
        for j in range(m):
            U_rows.append(i * 6 + j)
    
    V_rows = []
    for j in range(m):
        for k in range(p):
            V_rows.append(j * 6 + k)
    
    W_rows = []
    for k in range(p):
        for i in range(n):
            W_rows.append(k * 6 + i)
    
    U = full_U[U_rows, :]
    V = full_V[V_rows, :]
    W = full_W[W_rows, :]
    
    # Remove any all-zero columns (shouldn't happen, but safety)
    nonzero_cols = np.any(np.abs(U) > 1e-15, axis=0) & \
                   np.any(np.abs(V) > 1e-15, axis=0) & \
                   np.any(np.abs(W) > 1e-15, axis=0)
    U = U[:, nonzero_cols]
    V = V[:, nonzero_cols]
    W = W[:, nonzero_cols]
    
    rank = U.shape[1]
    return U, V, W, rank


def _verify_fast(U, V, W, n, m, p):
    """Fast verification using einsum.
    
    U has shape (n*m, r) with U[i*m+j, c]
    V has shape (m*p, r) with V[j*p+k, c]
    W has shape (p*n, r) with W[k*n+i, c]
    
    The tensor T[i,j,k] should equal 1 for all i,j,k.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)

# EVOLVE-BLOCK-END

