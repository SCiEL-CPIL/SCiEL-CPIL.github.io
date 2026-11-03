import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a low-rank tensor decomposition for 3x4 * 4x7 matrix multiplication.
    Uses block decomposition:
    n=3 -> [2, 1]
    m=4 -> [2, 2] (implicitly in Strassen for 2x2x2 blocks)
    p=7 -> [2, 2, 2, 1]
    
    Blocks:
    (2, 4, 2) -> 2 * Strassen(2,2,2) -> 14
    (2, 4, 1) -> 2 * Schoolbook(2,2,1) -> 8
    (1, 4, 2) -> 2 * Schoolbook(1,2,2) -> 8
    (1, 4, 1) -> 2 * Schoolbook(1,2,1) -> 4
    
    Total: 3*(14+8) + 3*(8) + 4 = 3*22 + 24 + 4 = 66 + 28 = 94? 
    Wait, let's re-calculate based on the p-split.
    p blocks: [0,1], [2,3], [4,5], [6].
    
    For n_block size 2 (rows 0,1):
      p_block size 2 (3 times): M_2,4,2. 
        Split m=4 into 2+2. Two M_2,2,2 blocks. Rank 7+7=14.
        3 * 14 = 42.
      p_block size 1 (1 time): M_2,4,1.
        Split m=4 into 2+2. Two M_2,2,1 blocks. Rank 4+4=8.
        1 * 8 = 8.
      Subtotal for n=2: 50.
      
    For n_block size 1 (row 2):
      p_block size 2 (3 times): M_1,4,2.
        Split m=4 into 2+2. Two M_1,2,2 blocks. Rank 4+4=8.
        3 * 8 = 24.
      p_block size 1 (1 time): M_1,4,1.
        Split m=4 into 2+2. Two M_1,2,1 blocks. Rank 2+2=4.
        1 * 4 = 4.
      Subtotal for n=1: 28.
      
    Total Rank: 50 + 28 = 78.
    """
    start_time = time.time()
    deadline = 140.0

    n, m, p = DIMENSIONS  # 3, 4, 7

    U_vecs = []
    V_vecs = []
    W_vecs = []

    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        u = [0.0] * (n * m)
        v = [0.0] * (m * p)
        w = [0.0] * (p * n)
        for idx, val in zip(u_indices, u_vals):
            u[idx] += val
        for idx, val in zip(v_indices, v_vals):
            v[idx] += val
        for idx, val in zip(w_indices, w_vals):
            w[idx] += val
        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)

    def strassen_222(i0, j0, k0):
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1
        # Standard Strassen 7 terms
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C01 += M1, C10 -= M1, C11 += M1
        # M2 = (A10+A11)B00 -> C10 += M2, C11 += M2
        # M3 = A00(B10-B11) -> C00 -= M3, C01 -= M3
        # M4 = A11(B01-B00) -> C00 += M4, C11 -= M4
        # M5 = (A00+A01)B11 -> C01 += M5, C11 += M5
        # M6 = (A10-A00)(B00+B01) -> C10 -= M6
        # M7 = (A01-A11)(B10+B11) -> C01 -= M7
        
        # Note: The standard Strassen formula maps to:
        # C00 = M1 + M4 - M3
        # C01 = M1 + M5 - M7
        # C10 = M2 - M6
        # C11 = M1 + M2 - M4 + M5
        
        # Let's verify the indices and signs carefully.
        # A is 2x2, B is 2x2, C is 2x2.
        # i0,i1 rows of A. j0,j1 cols of A / rows of B. k0,k1 cols of B / cols of C.
        
        # Term 1: (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        # Contributes to C[i0,k0] (+), C[i0,k1] (+), C[i1,k0] (-), C[i1,k1] (+)
        add_rank1_term(
            [i0*m+j0, i1*m+j1], [1, 1],
            [j0*p+k0, j1*p+k1], [1, 1],
            [k0*n+i0, k0*n+i1, k1*n+i0, k1*n+i1], [1, -1, 1, 1]
        )
        
        # Term 2: (A[i1,j0] + A[i1,j1]) * B[j0,k0]
        # Contributes to C[i1,k0] (+), C[i1,k1] (+)
        add_rank1_term(
            [i1*m+j0, i1*m+j1], [1, 1],
            [j0*p+k0], [1],
            [k0*n+i1, k1*n+i1], [1, 1]
        )
        
        # Term 3: A[i0,j0] * (B[j1,k0] - B[j1,k1])
        # Contributes to C[i0,k0] (-), C[i0,k1] (-)
        add_rank1_term(
            [i0*m+j0], [1],
            [j1*p+k0, j1*p+k1], [1, -1],
            [k0*n+i0, k1*n+i0], [-1, -1]
        )
        
        # Term 4: A[i1,j1] * (B[j0,k1] - B[j0,k0])
        # Contributes to C[i0,k0] (+), C[i1,k1] (-)
        add_rank1_term(
            [i1*m+j1], [1],
            [j0*p+k1, j0*p+k0], [1, -1],
            [k0*n+i0, k1*n+i1], [1, -1]
        )
        
        # Term 5: (A[i0,j0] + A[i0,j1]) * B[j1,k1]
        # Contributes to C[i0,k1] (+), C[i1,k1] (+)
        add_rank1_term(
            [i0*m+j0, i0*m+j1], [1, 1],
            [j1*p+k1], [1],
            [k1*n+i0, k1*n+i1], [1, 1]
        )
        
        # Term 6: (A[i1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k1])
        # Contributes to C[i1,k0] (-)
        add_rank1_term(
            [i1*m+j0, i0*m+j0], [1, -1],
            [j0*p+k0, j0*p+k1], [1, 1],
            [k0*n+i1], [-1]
        )
        
        # Term 7: (A[i0,j1] - A[i1,j1]) * (B[j1,k0] + B[j1,k1])
        # Contributes to C[i0,k1] (-)
        add_rank1_term(
            [i0*m+j1, i1*m+j1], [1, -1],
            [j1*p+k0, j1*p+k1], [1, 1],
            [k1*n+i0], [-1]
        )

    def schoolbook_block(i0, ni, j0, nj, k0, nk):
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    ii = i0 + i
                    jj = j0 + j
                    kk = k0 + k
                    add_rank1_term([ii*m+jj], [1.0],
                                   [jj*p+kk], [1.0],
                                   [kk*n+ii], [1.0])

    # Build decomposition
    if time.time() - start_time < deadline:
        # n=0,1 (size 2)
        # p=0,1 (size 2)
        strassen_222(0, 0, 0)
        strassen_222(0, 2, 0)
        # p=2,3 (size 2)
        strassen_222(0, 0, 2)
        strassen_222(0, 2, 2)
        # p=4,5 (size 2)
        strassen_222(0, 0, 4)
        strassen_222(0, 2, 4)
        # p=6 (size 1)
        schoolbook_block(0, 2, 0, 4, 6, 1)
        
        # n=2 (size 1)
        # p=0,1 (size 2)
        schoolbook_block(2, 1, 0, 4, 0, 2)
        # p=2,3 (size 2)
        schoolbook_block(2, 1, 0, 4, 2, 2)
        # p=4,5 (size 2)
        schoolbook_block(2, 1, 0, 4, 4, 2)
        # p=6 (size 1)
        schoolbook_block(2, 1, 0, 4, 6, 1)

    rank = len(U_vecs)
    
    U = np.array(U_vecs).T
    V = np.array(V_vecs).T
    W = np.array(W_vecs).T

    verified = True
    if time.time() - start_time < deadline - 5.0:
        # Vectorized verification
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # We want T[i,j,k] == 1 for all i,j,k.
        
        # U shape: (n*m, rank)
        # V shape: (m*p, rank)
        # W shape: (p*n, rank)
        
        # Reshape to 3D tensors for easier indexing?
        # U_3d[i, j, r]
        # V_3d[j, k, r]
        # W_3d[k, i, r]
        
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        # Compute sum over r
        # T[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
        
        # We can loop over i, j, k and check.
        # Or use einsum.
        # T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        # Check if T is all 1s
        if not np.allclose(T, 1.0, atol=1e-9):
            verified = False

    if not verified:
        # Fallback to schoolbook
        rank_school = n * m * p
        U_sb = [[0] * rank_school for _ in range(n * m)]
        V_sb = [[0] * rank_school for _ in range(m * p)]
        W_sb = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
