"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    deadline = time.time() + 115.0  # safety margin before 150s kill

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition."""
        U = [
            [1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 0, 1],
        ]
        V = [
            [1, 0, 0, 0, 0, 1, 1],
            [0, 1, -1, 1, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0],
            [1, 1, 0, -1, 1, 0, 0],
        ]
        W = [
            [1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0, 1, 0],
            [1, 0, 0, 1, -1, 0, 0],
        ]
        return U, V, W

    def verify_fast(U, V, W, n, m, p, tol=1e-9):
        """Faster verification using einsum."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        return np.allclose(T, np.ones((n, m, p)), atol=tol, rtol=0)

    # Start with schoolbook baseline
    best_result = schoolbook()
    best_rank = n * m * p

    # Build a decomposition using Strassen blocks for the 2x2 submatrices
    # and rank-1 terms for the remaining parts.
    
    U_cols = []
    V_cols = []
    W_cols = []

    def add_strassen_block(i_off, j_off, k_off):
        """Add 7 Strassen terms for 2x2x2 block at given offsets."""
        U_s, V_s, W_s = strassen_2x2x2()
        for c in range(7):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            for ii in range(2):
                for jj in range(2):
                    U_col[(i_off+ii)*m + (j_off+jj)] = U_s[ii*2+jj][c]
            for jj in range(2):
                for kk in range(2):
                    V_col[(j_off+jj)*p + (k_off+kk)] = V_s[jj*2+kk][c]
            for kk in range(2):
                for ii in range(2):
                    W_col[(k_off+kk)*n + (i_off+ii)] = W_s[kk*2+ii][c]
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

    def add_rank1(i, j, k):
        """Add a single rank-1 term: U[i,j]=1, V[j,k]=1, W[k,i]=1."""
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        U_col[i*m + j] = 1.0
        V_col[j*p + k] = 1.0
        W_col[k*n + i] = 1.0
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    # For i in {0,1}, j in {0,1}: use Strassen for k={0,1} and k={2,3}
    add_strassen_block(0, 0, 0)  # i={0,1}, j={0,1}, k={0,1}
    add_strassen_block(0, 0, 2)  # i={0,1}, j={0,1}, k={2,3}
    
    # For i in {0,1}, j in {0,1}, k=4: use 1 rank-1 term
    # T[0,j,4]=1, T[1,j,4]=1 for j in {0,1}
    # We can use: U[i,j,c]=1 for i in {0,1}, j in {0,1}; V[j,4,c]=1 for j in {0,1}; W[4,i,c]=1 for i in {0,1}
    # But wait, this would contribute to other k values if V[j,k,c] is non-zero for k!=4.
    # We need V[j,k,c] = delta_{k,4} * 1, W[k,i,c] = delta_{k,4} * 1
    # So: U[0,0,c]=1, U[0,1,c]=1, U[1,0,c]=1, U[1,1,c]=1
    #     V[0,4,c]=1, V[1,4,c]=1
    #     W[4,0,c]=1, W[4,1,c]=1
    U_col = [0.0] * (n * m)
    V_col = [0.0] * (m * p)
    W_col = [0.0] * (p * n)
    for i in [0, 1]:
        for j in [0, 1]:
            U_col[i*m + j] = 1.0
    for j in [0, 1]:
        V_col[j*p + 4] = 1.0
    for i in [0, 1]:
        W_col[4*n + i] = 1.0
    U_cols.append(U_col)
    V_cols.append(V_col)
    W_cols.append(W_col)

    # For i in {0,1}, j in {2,3}: use Strassen for k={0,1} and k={2,3}
    add_strassen_block(0, 2, 0)  # i={0,1}, j={2,3}, k={0,1}
    add_strassen_block(0, 2, 2)  # i={0,1}, j={2,3}, k={2,3}
    
    # For i in {0,1}, j in {2,3}, k=4: use 1 rank-1 term
    U_col = [0.0] * (n * m)
    V_col = [0.0] * (m * p)
    W_col = [0.0] * (p * n)
    for i in [0, 1]:
        for j in [2, 3]:
            U_col[i*m + j] = 1.0
    for j in [2, 3]:
        V_col[j*p + 4] = 1.0
    for i in [0, 1]:
        W_col[4*n + i] = 1.0
    U_cols.append(U_col)
    V_cols.append(V_col)
    W_cols.append(W_col)

    # For i=2: all j in {0,1,2,3}, all k in {0,1,2,3,4}
    # T[2,j,k]=1 for all j,k. This is a 4x5 all-ones matrix.
    # We can use: U[2,j,c]=1 for all j; V[j,k,c]=1 for all j,k; W[k,2,c]=1 for all k
    # This gives one term that covers all of i=2.
    U_col = [0.0] * (n * m)
    V_col = [0.0] * (m * p)
    W_col = [0.0] * (p * n)
    for j in range(m):
        U_col[2*m + j] = 1.0
    for j in range(m):
        for k in range(p):
            V_col[j*p + k] = 1.0
    for k in range(p):
        W_col[k*n + 2] = 1.0
    U_cols.append(U_col)
    V_cols.append(V_col)
    W_cols.append(W_col)

    rank = len(U_cols)
    U_mat = [[U_cols[c][r] for c in range(rank)] for r in range(n*m)]
    V_mat = [[V_cols[c][r] for c in range(rank)] for r in range(m*p)]
    W_mat = [[W_cols[c][r] for c in range(rank)] for r in range(p*n)]

    if verify_fast(U_mat, V_mat, W_mat, n, m, p):
        if rank < best_rank:
            best_result = {"U": U_mat, "V": V_mat, "W": W_mat}
            best_rank = rank

    # Try to further optimize: for i in {0,1}, k=4, the two rank-1 terms above
    # can be merged into one if we extend U to cover all j in {0,1,2,3}
    # But we already have separate j groups. Let's try a different approach:
    # Use one term for all i in {0,1}, all j, k=4
    # T[i,j,4]=1 for i in {0,1}, j in {0,1,2,3}
    # U[i,j,c]=1 for i in {0,1}, all j; V[j,4,c]=1 for all j; W[4,i,c]=1 for i in {0,1}
    
    # Let's build this alternative
    U_cols2 = []
    V_cols2 = []
    W_cols2 = []

    # Strassen blocks (same as before)
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)

    # For i in {0,1}, k=4: use 1 term instead of 2
    U_col = [0.0] * (n * m)
    V_col = [0.0] * (m * p)
    W_col = [0.0] * (p * n)
    for i in [0, 1]:
        for j in range(m):
            U_col[i*m + j] = 1.0
    for j in range(m):
        V_col[j*p + 4] = 1.0
    for i in [0, 1]:
        W_col[4*n + i] = 1.0
    U_cols2.append(U_col)
    V_cols2.append(V_col)
    W_cols2.append(W_col)

    # For i=2: use 1 term
    U_col = [0.0] * (n * m)
    V_col = [0.0] * (m * p)
    W_col = [0.0] * (p * n)
    for j in range(m):
        U_col[2*m + j] = 1.0
    for j in range(m):
        for k in range(p):
            V_col[j*p + k] = 1.0
    for k in range(p):
        W_col[k*n + 2] = 1.0
    U_cols2.append(U_col)
    V_cols2.append(V_col)
    W_cols2.append(W_col)

    rank2 = len(U_cols2)
    U_mat2 = [[U_cols2[c][r] for c in range(rank2)] for r in range(n*m)]
    V_mat2 = [[V_cols2[c][r] for c in range(rank2)] for r in range(m*p)]
    W_mat2 = [[W_cols2[c][r] for c in range(rank2)] for r in range(p*n)]

    if verify_fast(U_mat2, V_mat2, W_mat2, n, m, p):
        if rank2 < best_rank:
            best_result = {"U": U_mat2, "V": V_mat2, "W": W_mat2}
            best_rank = rank2

    return best_result
# EVOLVE-BLOCK-END
