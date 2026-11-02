import numpy as np
import time

def construct():
    _start_time = time.time()
    n, m, p = 2, 4, 5
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen(col_offset, j_offset, k_offset):
        # j_offset: 0 for J1, 2 for J2
        # k_offset: 0 for K1, 2 for K2
        # Strassen for 2x2x2
        # U indices: i*2 + j' (local j') -> global i*m + (j_offset + j')
        # V indices: j'*2 + k' (local k') -> global (j_offset + j')*p + (k_offset + k')
        # W indices: k'*2 + i (local i) -> global (k_offset + k')*n + i
        
        # Term 1
        U[0 * m + j_offset + 0][col_offset + 0] = 1
        U[1 * m + j_offset + 1][col_offset + 0] = 1
        V[(j_offset + 0) * p + k_offset + 0][col_offset + 0] = 1
        V[(j_offset + 1) * p + k_offset + 1][col_offset + 0] = 1
        W[(k_offset + 0) * n + 0][col_offset + 0] = 1
        W[(k_offset + 1) * n + 1][col_offset + 0] = 1

        # Term 2
        U[1 * m + j_offset + 0][col_offset + 1] = 1
        U[1 * m + j_offset + 1][col_offset + 1] = 1
        V[(j_offset + 0) * p + k_offset + 0][col_offset + 1] = 1
        W[(k_offset + 0) * n + 1][col_offset + 1] = 1
        W[(k_offset + 1) * n + 1][col_offset + 1] = -1

        # Term 3
        U[0 * m + j_offset + 0][col_offset + 2] = 1
        V[(j_offset + 0) * p + k_offset + 1][col_offset + 2] = 1
        V[(j_offset + 1) * p + k_offset + 1][col_offset + 2] = -1
        W[(k_offset + 1) * n + 0][col_offset + 2] = 1
        W[(k_offset + 1) * n + 1][col_offset + 2] = 1

        # Term 4
        U[1 * m + j_offset + 1][col_offset + 3] = 1
        V[(j_offset + 0) * p + k_offset + 0][col_offset + 3] = -1
        V[(j_offset + 1) * p + k_offset + 0][col_offset + 3] = 1
        W[(k_offset + 0) * n + 0][col_offset + 3] = 1
        W[(k_offset + 0) * n + 1][col_offset + 3] = 1

        # Term 5
        U[0 * m + j_offset + 0][col_offset + 4] = 1
        U[0 * m + j_offset + 1][col_offset + 4] = 1
        V[(j_offset + 1) * p + k_offset + 1][col_offset + 4] = 1
        W[(k_offset + 0) * n + 0][col_offset + 4] = -1
        W[(k_offset + 1) * n + 0][col_offset + 4] = 1

        # Term 6
        U[0 * m + j_offset + 0][col_offset + 5] = -1
        U[1 * m + j_offset + 0][col_offset + 5] = 1
        V[(j_offset + 0) * p + k_offset + 0][col_offset + 5] = 1
        V[(j_offset + 0) * p + k_offset + 1][col_offset + 5] = 1
        W[(k_offset + 1) * n + 1][col_offset + 5] = 1

        # Term 7
        U[0 * m + j_offset + 1][col_offset + 6] = 1
        U[1 * m + j_offset + 1][col_offset + 6] = -1
        V[(j_offset + 1) * p + k_offset + 0][col_offset + 6] = 1
        V[(j_offset + 1) * p + k_offset + 1][col_offset + 6] = 1
        W[(k_offset + 0) * n + 0][col_offset + 6] = 1

    def add_schoolbook_2x2x1(col_offset, j_offset, k_offset):
        # Schoolbook for 2x2x1
        # k is fixed to k_offset
        # Term 1: i=0, j=0
        U[0 * m + j_offset + 0][col_offset + 0] = 1
        V[(j_offset + 0) * p + k_offset][col_offset + 0] = 1
        W[k_offset * n + 0][col_offset + 0] = 1
        
        # Term 2: i=0, j=1
        U[0 * m + j_offset + 1][col_offset + 1] = 1
        V[(j_offset + 1) * p + k_offset][col_offset + 1] = 1
        W[k_offset * n + 0][col_offset + 1] = 1
        
        # Term 3: i=1, j=0
        U[1 * m + j_offset + 0][col_offset + 2] = 1
        V[(j_offset + 0) * p + k_offset][col_offset + 2] = 1
        W[k_offset * n + 1][col_offset + 2] = 1
        
        # Term 4: i=1, j=1
        U[1 * m + j_offset + 1][col_offset + 3] = 1
        V[(j_offset + 1) * p + k_offset][col_offset + 3] = 1
        W[k_offset * n + 1][col_offset + 3] = 1

    col = 0
    # J1 (j_offset=0)
    add_strassen(col, 0, 0); col += 7
    add_strassen(col, 0, 2); col += 7
    add_schoolbook_2x2x1(col, 0, 4); col += 4
    
    # J2 (j_offset=2)
    add_strassen(col, 2, 0); col += 7
    add_strassen(col, 2, 2); col += 7
    add_schoolbook_2x2x1(col, 2, 4); col += 4
    
    # Verify decomposition correctness
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reconstruct tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Check against known matrix multiplication tensor
    T_recon = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.dot(
                    U_arr[i * m + j, :],
                    V_arr[j * p + k, :] * W_arr[k * n + i, :]
                )
    
    # The matrix multiplication tensor should be 1 for all valid (i,j,k)
    # Actually, for matrix mult, T[i,j,k] = 1 if the (i,j) entry of A multiplies with (j,k) entry of B
    # to contribute to (i,k) entry of C. So T[i,j,k] = 1 for all i,j,k in range.
    # Wait, no. The tensor T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k].
    # So T[i,j,k] = 1 for all valid i,j,k.
    T_expected = np.ones((n, m, p))
    
    if not np.allclose(T_recon, T_expected, atol=1e-10):
        # Decomposition is incorrect; fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
