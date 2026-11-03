import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    def check_validity(U, V, W, tol=1e-6):
        R = U.shape[1]
        U_mat = U.reshape(n, m, R)
        V_mat = V.reshape(m, p, R)
        W_mat = W.reshape(p, n, R)
        err = 0.0
        for i in range(n):
            for k in range(p):
                M_ik = np.zeros((m, m))
                for r in range(R):
                    M_ik += np.outer(U_mat[i, :, r] * W_mat[k, i, r], V_mat[:, k, r])
                err += np.sum((M_ik - np.eye(m))**2)
        return err

    def strassen_2x2_local():
        """Returns (U, V, W) for 2x2x2 multiplication. Shapes (4,7), (4,7), (4,7)."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Indices: u[i, j], v[j, k], w[k, i]
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        
        # M1 = (c+d)(e+f)
        U[2, 0] = 1; U[3, 0] = 1  # c, d
        V[0, 0] = 1; V[1, 0] = 1  # e, f
        W[0, 0] = 1; W[1, 0] = 1  # c00, c01? No, W is (p*n, R). p=2, n=2.
        # W index: k*n + i.
        # M1 contributes to C00 and C01?
        # Strassen:
        # C00 = M1 + M4 - M2 - M7
        # C01 = M3 + M5
        # C10 = M2 + M6
        # C11 = M1 + M3 - M4 - M5
        
        # Let's map W indices:
        # W[0] -> C00 (k=0, i=0)
        # W[1] -> C01 (k=1, i=0)
        # W[2] -> C10 (k=0, i=1)
        # W[3] -> C11 (k=1, i=1)
        
        # M1
        W[0, 0] = 1; W[1, 0] = 1  # C00, C01? 
        # Wait, standard Strassen:
        # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
        # M2 = A22(B21-B11) -> C21 += M2
        # M3 = (A11-A22)(B12+B22) -> C12 += M3
        # M4 = A11(B11-B22) -> C11 += M4
        # M5 = (A11+A21)B22 -> C22 += M5
        # M6 = (A22-A12)B11 -> C21 += M6
        # M7 = (A12-A21)(B11+B12) -> C12 += M7
        
        # My indices:
        # A11=A[0,0], A12=A[0,1], A21=A[1,0], A22=A[1,1]
        # B11=B[0,0], B12=B[0,1], B21=B[1,0], B22=B[1,1]
        # C11=C[0,0], C12=C[0,1], C21=C[1,0], C22=C[1,1]
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1  # A11, A22
        V[0, 0] = 1; V[3, 0] = 1  # B11, B22
        W[0, 0] = 1; W[3, 0] = 1  # C11, C22
        
        # M2
        U[3, 1] = 1  # A22
        V[2, 1] = 1; V[0, 1] = -1  # B21, -B11
        W[2, 1] = 1  # C21
        
        # M3
        U[0, 2] = 1; U[3, 2] = -1  # A11, -A22
        V[1, 2] = 1; V[3, 2] = 1  # B12, B22
        W[1, 2] = 1  # C12
        
        # M4
        U[0, 3] = 1  # A11
        V[0, 3] = 1; V[3, 3] = -1  # B11, -B22
        W[0, 3] = 1  # C11
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1  # A11, A21
        V[3, 4] = 1  # B22
        W[3, 4] = 1  # C22
        
        # M6
        U[3, 5] = 1; U[1, 5] = -1  # A22, -A12
        V[0, 5] = 1  # B11
        W[2, 5] = 1  # C21
        
        # M7
        U[1, 6] = 1; U[2, 6] = -1  # A12, -A21
        V[0, 6] = 1; V[1, 6] = 1  # B11, B12
        W[1, 6] = 1  # C12
        
        return U, V, W

    def trivial_2x1x2_local():
        """Returns (U, V, W) for 2x1x2 multiplication. Shapes (2,4), (2,4), (4,4)."""
        # A is 2x1 (col j=2 of A). B is 1x2 (row j=2 of B).
        # C is 2x2.
        # C_ik = A_i B_k
        U = np.zeros((2, 4))
        V = np.zeros((2, 4))
        W = np.zeros((4, 4))
        
        # 4 terms
        for k in range(2):
            for i in range(2):
                r = k * 2 + i
                U[i, r] = 1
                V[k, r] = 1
                W[k * n + i, r] = 1
        
        return U, V, W

    def optimal_2x3x2_block(k0, k1):
        """
        Returns (U_block, V_block, W_block) for 2x3x2 multiplication.
        U_block shape (6, 11), V_block shape (6, 11), W_block shape (4, 11).
        """
        U_b = np.zeros((n * m, 11))
        V_b = np.zeros((m * 2, 11))
        W_b = np.zeros((2 * n, 11))
        
        # 1. Strassen part for j in {0, 1}
        U_s, V_s, W_s = strassen_2x2_local()
        
        # Map U_s (4x7) to U_b (6x11)
        # U_s rows 0,1,2,3 correspond to A[0,0], A[0,1], A[1,0], A[1,1]
        # U_b rows 0,1,2,3 correspond to A[0,0], A[0,1], A[1,0], A[1,1]
        U_b[0:4, 0:7] = U_s
        
        # Map V_s (4x7) to V_b (6x11)
        # V_s rows 0,1,2,3 correspond to B[0,0], B[0,1], B[1,0], B[1,1]
        # V_b rows 0,1,2,3 correspond to B[0,0], B[0,1], B[1,0], B[1,1]
        # Note: V_b is (m*2, 11) = (6, 11).
        # V_b row index = j * 2 + kk.
        # B[0,0] -> j=0, kk=0 -> 0
        # B[0,1] -> j=0, kk=1 -> 1
        # B[1,0] -> j=1, kk=0 -> 2
        # B[1,1] -> j=1, kk=1 -> 3
        V_b[0:4, 0:7] = V_s
        
        # Map W_s (4x7) to W_b (4x11)
        # W_s rows 0,1,2,3 correspond to C[0,0], C[0,1], C[1,0], C[1,1]
        # W_b rows 0,1,2,3 correspond to C[0,0], C[0,1], C[1,0], C[1,1]
        # W_b row index = kk * n + i.
        # C[0,0] -> kk=0, i=0 -> 0
        # C[0,1] -> kk=1, i=0 -> 1
        # C[1,0] -> kk=0, i=1 -> 2
        # C[1,1] -> kk=1, i=1 -> 3
        W_b[0:4, 0:7] = W_s
        
        # 2. Trivial part for j = 2
        U_t, V_t, W_t = trivial_2x1x2_local()
        
        # Map U_t (2x4) to U_b (6x11)
        # U_t rows 0,1 correspond to A[0,2], A[1,2]
        # U_b rows 4,5 correspond to A[0,2], A[1,2]
        U_b[4:6, 7:11] = U_t
        
        # Map V_t (2x4) to V_b (6x11)
        # V_t rows 0,1 correspond to B[2,0], B[2,1]
        # V_b rows 4,5 correspond to B[2,0], B[2,1]
        # V_b row index = j * 2 + kk.
        # B[2,0] -> j=2, kk=0 -> 4
        # B[2,1] -> j=2, kk=1 -> 5
        V_b[4:6, 7:11] = V_t
        
        # Map W_t (4x4) to W_b (4x11)
        # W_t rows 0,1,2,3 correspond to C[0,0], C[0,1], C[1,0], C[1,1]
        # W_b rows 0,1,2,3 correspond to C[0,0], C[0,1], C[1,0], C[1,1]
        W_b[0:4, 7:11] = W_t
        
        return U_b, V_b, W_b

    def trivial_block(k_start, k_end):
        width = k_end - k_start
        rank = n * m * width
        U_b = np.zeros((n * m, rank))
        V_b = np.zeros((m * width, rank))
        W_b = np.zeros((width * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for kk in range(width):
                    U_b[i * m + j, col] = 1
                    V_b[j * width + kk, col] = 1
                    W_b[kk * n + i, col] = 1
                    col += 1
        return U_b, V_b, W_b

    def combine_blocks(blocks):
        total_rank = sum(U_b.shape[1] for U_b, _, _, _, _ in blocks)
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        col_offset = 0
        for U_b, V_b, W_b, k_start, k_end in blocks:
            r = U_b.shape[1]
            U[:, col_offset:col_offset + r] = U_b
            V_b_full = np.zeros((m * p, r))
            V_b_full[k_start * m:(k_end) * m, :] = V_b
            V[:, col_offset:col_offset + r] = V_b_full
            W_b_full = np.zeros((p * n, r))
            W_b_full[k_start * n:(k_end) * n, :] = W_b
            W[:, col_offset:col_offset + r] = W_b_full
            col_offset += r
        return U, V, W

    # Use 5 blocks of size 2
    blocks = []
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        U_b, V_b, W_b = optimal_2x3x2_block(k0, k1)
        blocks.append((U_b, V_b, W_b, k0, k1))
        
    U, V, W = combine_blocks(blocks)
    
    # Verify
    err = check_validity(U, V, W)
    if err > 1e-6:
        # Fallback to trivial
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
