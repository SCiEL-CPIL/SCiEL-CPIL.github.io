"""Tensor decomposition for 2x2 * 2x13 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120

    n, m, p = 2, 2, 13

    def _build_schoolbook():
        rank_sb = n * m * p
        U = np.zeros((n * m, rank_sb))
        V = np.zeros((m * p, rank_sb))
        W = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k, col] = 1.0
                    V[k * p + j, col] = 1.0
                    W[j * n + i, col] = 1.0
                    col += 1
        return U, V, W

    def _build_strassen_6x7_4():
        # 6 Strassen 2x2x2x2 blocks (rank 7 each) + 1 schoolbook 2x2x2x1 (rank 4)
        rank = 6 * 7 + 4
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        strassen_terms = [
            (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
            (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1])),
            (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1])),
            (np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0])),
            (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0])),
            (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])),
            (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0])),
        ]

        col = 0
        for block in range(6):
            j0 = block * 2
            j1 = block * 2 + 1
            for t in range(7):
                u_loc, v_loc, w_loc = strassen_terms[t]
                U[0, col] = u_loc[0]
                U[1, col] = u_loc[1]
                U[2, col] = u_loc[2]
                U[3, col] = u_loc[3]
                V[0 * p + j0, col] = v_loc[0]
                V[0 * p + j1, col] = v_loc[1]
                V[1 * p + j0, col] = v_loc[2]
                V[1 * p + j1, col] = v_loc[3]
                W[j0 * n + 0, col] = w_loc[0]
                W[j0 * n + 1, col] = w_loc[1]
                W[j1 * n + 0, col] = w_loc[2]
                W[j1 * n + 1, col] = w_loc[3]
                col += 1

        j_last = 12
        for i in range(n):
            for k in range(m):
                U[i * m + k, col] = 1.0
                V[k * p + j_last, col] = 1.0
                W[j_last * n + i, col] = 1.0
                col += 1
        return U, V, W

    def _build_polynomial_42():
        # 3 polynomial blocks over truncated dual numbers Z[x]/(x^5): rank 42
        # B columns 0..4, 5..9, 10..12 (last block has only 3 columns)
        rank = 42
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        # Polynomial multiplication coefficients for degree < 5 (5 terms: x^0..x^4)
        # Product of (a0 + a1*x + a2*x^2) * (b0 + b1*x + b2*x^2 + b3*x^3 + b4*x^4)
        # We need c0..c4 where c_k = sum_{i=0}^{k} a_i * b_{k-i}, truncated at degree 4
        # Since A is 2x2 with entries a_00, a_01, a_10, a_11
        # A row 0: [a00, a01], row 1: [a10, a11]
        # B row k: b_k (a vector of 5 coefficients for k=0,1)
        # C = A * B, C[i][j] = sum_k A[i][k] * B[k][j]
        
        # We treat each block of B as a 2x5 matrix where rows are the 2 matrix rows of B
        # and columns are the 5 polynomial coefficients.
        # For a single block, we need to compute A(2x2) * B_block(2x5) = C_block(2x5)
        # C_block[i][c] = A[i][0]*B[0][c] + A[i][1]*B[1][c]
        
        # This is just bilinear: for each i, c, the term is A[i,0]*B[0,c] + A[i,1]*B[1,c]
        # Schoolbook for 2x2 * 2x5 has rank 20. But we can use polynomial structure.
        
        # Actually, for a fixed A(2x2), the map B(2x5) -> C(2x5) is:
        # C[i][c] = A[i][0]*B[0][c] + A[i][1]*B[1][c]
        # This is just: for each c, C[:,c] = A * B[:,c]
        # That's 5 independent 2x2*1x1 products (scalar multiplies of rows of A).
        # Wait, that's not right. A is 2x2, B[:,c] is a 2-vector. C[:,c] = A * B[:,c] is 2x1.
        # So for each column c of B, we do a 2x2 times 2x1 product = 2x1 result.
        # That's rank 4 per column, 5 columns = rank 20 per block. 3 blocks * 20 = 60.
        # That's worse than 46. Let me reconsider.
        
        # The polynomial approach: treat B as a matrix over the ring Z[x]/(x^13).
        # A is a 2x2 matrix over Z.
        # C = A * B where the multiplication is in the ring.
        # A[i][k] * B[k][j] = A[i][k] * (sum_{t} b[k][j][t] * x^t)
        # C[i][j][s] = sum_{t=0}^{s} (sum_k A[i][k] * b[k][j][s-t])
        
        # So C[i][j][s] = sum_k A[i][k] * (sum_{t=0}^{s} b[k][j][s-t])
        # = sum_k A[i][k] * conv(b[k][j], [1,1,1,1,1,1,1,1,1,1,1,1,1])[s]
        
        # This is a convolution. The rank of the convolution tensor for length 13 is 13.
        # For each (i,k) pair and each output position s, we need the convolution.
        # The bilinear map has structure: C[i,j,s] = sum_k A[i,k] * (B[k,j] * x^s coefficient)
        
        # Let me think of it differently. The tensor T[i*m+k, k*p+j, j*n+i] = 1.
        # We can group by (i,k) pairs. For fixed i,k, the slice is T[i*m+k, k*p+j, j*n+i] = 1 for all j.
        # This is a rank-1 slice in the sense that U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] sums to 1.
        
        # Actually, the simplest improvement: use the fact that for fixed i and k,
        # the dependence on j is trivial (it's just 1 for all j).
        # So U[i*m+k, col] can be 1, V[k*p+j, col] can be 1, W[j*n+i, col] can be 1.
        # That's the schoolbook decomposition with rank 52.
        
        # Strassen gives 46. Let me just return a rank-46 construction.
        # For now, return the Strassen-based construction.
        return _build_strassen_6x7_4()

    # Build schoolbook (rank 52)
    U_school, V_school, W_school = _build_schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_school, V_school, W_school

    # Build Strassen-based (rank 46)
    try:
        U_46, V_46, W_46 = _build_strassen_6x7_4()
        # Verify
        T_target = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    T_target[i * m + k, k * p + j, j * n + i] = 1.0
        T_approx = np.einsum('it,jt,kt->ijk', U_46, V_46, W_46)
        error = np.max(np.abs(T_approx - T_target))
        if error < 1e-10:
            best_rank = 46
            best_U, best_V, best_W = U_46, V_46, W_46
    except Exception:
        pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
