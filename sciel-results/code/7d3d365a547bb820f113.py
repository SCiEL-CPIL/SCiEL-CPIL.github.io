import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target rank
    target_rank = 35
    
    # Deadline guard
    start_time = time.time()
    deadline = 100 # 100 seconds
    
    # Function to verify a decomposition
    def verify(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need T[i, j, k] = 1 if i,j,k correspond to a valid multiplication term?
        # No, T is the multiplication tensor.
        # T[i, j, k] = 1 if the term A_ij * B_jk contributes to C_ik?
        # C_ik = sum_j A_ij B_jk.
        # So T[i, j, k] = 1 for all i, j, k?
        # Wait.
        # A is n x m. B is m x p. C is n x p.
        # C_ik = sum_j A_ij B_jk.
        # The tensor T is in R^(n*m) x R^(m*p) x R^(n*p).
        # T[I, J, K] where I = i*m+j, J = j*p+k, K = i*p+k?
        # No, the task says W[k*n+i, r].
        # So K = k*n+i.
        # Let's check the indices.
        # U index: i*m + j. (i in 0..n-1, j in 0..m-1)
        # V index: j*p + k. (j in 0..m-1, k in 0..p-1)
        # W index: k*n + i. (k in 0..p-1, i in 0..n-1)
        
        # The tensor entry T[I, J, K] is non-zero if I, J, K are compatible.
        # I = i*m + j.
        # J = j*p + k.
        # K = k*n + i.
        # For a fixed i, j, k, we have I, J, K.
        # The term is A_ij * B_jk * C_ik.
        # So T[I, J, K] = 1.
        # All other entries are 0.
        
        R = U.shape[1]
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    I = i*m + j
                    J = j*p + k
                    K = k*n + i
                    T_target[I, J, K] = 1
        
        # Compute T_est
        T_est = np.zeros((n*m, m*p, p*n))
        for r in range(R):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            T_est += np.einsum('i,j,k->ijk', u, v, w)
            
        return np.allclose(T_est, T_target, atol=1e-9)

    # Generate the known rank 36 solution
    def get_rank_36_solution():
        rank = 36
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Part 1: 2x4x4 (Rank 28)
        # Split A into A1 (2x2), A2 (2x2)
        # Split B into B1 (2x4), B2 (2x4)
        # C = A1 B1 + A2 B2
        # A1 B1 is 2x2x4. Two 2x2x2.
        # A2 B2 is 2x2x4. Two 2x2x2.
        
        # We need to map Strassen's 2x2x2 decomposition.
        # Strassen:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B11 - B12)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We will fill U, V, W for these 7 terms for each 2x2x2 block.
        
        def add_strassen_block(offset_u, offset_v, offset_w, block_id):
            # block_id: 0 for A1B1_col1, 1 for A1B1_col2, 2 for A2B2_col1, 3 for A2B2_col2
            # This is getting complicated to map indices.
            # Let's just use the standard schoolbook for the 2x2x2 blocks?
            # No, we need rank 7.
            
            # Let's hardcode the Strassen coefficients.
            # A indices: 0,1,2,3 (00, 01, 10, 11)
            # B indices: 0,1,2,3
            # C indices: 0,1,2,3
            
            # U factors for A
            u_coeffs = [
                [1, 0, 0, 1], # M1
                [0, 0, 1, 1], # M2
                [1, 0, 0, 0], # M3
                [0, 0, 0, 1], # M4
                [1, 1, 0, 0], # M5
                [0, 0, 1, -1],# M6
                [0, 1, 0, -1] # M7
            ]
            
            # V factors for B
            v_coeffs = [
                [1, 0, 0, 1], # M1
                [1, 0, 0, 0], # M2
                [0, 1, 0, -1],# M3
                [1, -1, 0, 0],# M4
                [0, 0, 0, 1], # M5
                [1, 1, 0, 0], # M6
                [0, 0, 1, 1]  # M7
            ]
            
            # W factors for C
            w_coeffs = [
                [1, 0, 0, -1],# M1
                [0, 0, 1, 0], # M2
                [0, 1, 0, 0], # M3
                [1, 0, 1, 0], # M4
                [-1, 1, 0, 0],# M5
                [0, 0, 0, 1], # M6
                [1, 0, 0, 0]  # M7
            ]
            
            # Map these to the global indices.
            # This requires knowing which 2x2 block we are in.
            # For A1B1:
            # A1 is A[:, 0:2]. Indices in U: 0,1 (i=0, j=0,1) and 2,3 (i=1, j=0,1).
            # B1 is B[0:2, :]. Indices in V: 0..3 (j=0, k=0..1) and 4..7 (j=1, k=0..1).
            # C1 is C[:, 0:2]. Indices in W: 0,1 (k=0, i=0,1) and 2,3 (k=1, i=0,1).
            
            # This is too error-prone to do manually in code generation.
            # I will return the schoolbook rank 40 if I can't get 36 right quickly.
            # But I know 36 is possible.
            pass

    # Since I can't easily construct the 36 solution from scratch in a short time,
    # and the prompt says "score 1.0000" is the current best (which is 40),
    # and "1.111" was achieved by someone else (which is 36),
    # I should try to return the 36 solution.
    
    # If I return 40, I score 1.0.
    # If I return 36, I score 1.11.
    
    # Let's try to construct the 36 solution carefully.
    
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Part 1: 2x4x4 (Rank 28)
    # A = [A1, A2] (2x2 each)
    # B = [B1; B2] (2x4 each)
    # C = A1 B1 + A2 B2
    
    # A1 B1: 2x2x4. Split B1 into B1a, B1b (2x2 each).
    # A1 B1a (2x2x2) and A1 B1b (2x2x2).
    # A2 B2: 2x2x4. Split B2 into B2a, B2b (2x2 each).
    # A2 B2a (2x2x2) and A2 B2b (2x2x2).
    
    # Total 4 blocks of 2x2x2.
    # Each uses 7 terms.
    # Total 28 terms.
    
    # Indices:
    # A1: rows 0,1 of A. Cols 0,1 of A.
    # A2: rows 0,1 of A. Cols 2,3 of A.
    
    # B1: rows 0,1 of B. Cols 0..3 of B.
    # B2: rows 2,3 of B. Cols 0..3 of B.
    
    # C1: Cols 0..3 of C.
    
    # Block 1: A1 B1a. A1 (cols 0,1), B1a (rows 0,1, cols 0,1).
    # U indices: 0,1 (i=0, j=0,1) and 2,3 (i=1, j=0,1).
    # V indices: 0,1 (j=0, k=0,1) and 2,3 (j=1, k=0,1).
    # W indices: 0,1 (k=0, i=0,1) and 2,3 (k=1, i=0,1).
    
    # Block 2: A1 B1b. A1 (cols 0,1), B1b (rows 0,1, cols 2,3).
    # U indices: 0,1,2,3.
    # V indices: 4,5 (j=0, k=2,3) and 6,7 (j=1, k=2,3).
    # W indices: 4,5 (k=2, i=0,1) and 6,7 (k=3, i=0,1).
    
    # Block 3: A2 B2a. A2 (cols 2,3), B2a (rows 2,3, cols 0,1).
    # U indices: 4,5 (i=0, j=2,3) and 6,7 (i=1, j=2,3).
    # V indices: 8,9 (j=2, k=0,1) and 10,11 (j=3, k=0,1).
    # W indices: 0,1,2,3.
    
    # Block 4: A2 B2b. A2 (cols 2,3), B2b (rows 2,3, cols 2,3).
    # U indices: 4,5,6,7.
    # V indices: 12,13,14,15.
    # W indices: 4,5,6,7.
    
    def fill_strassen(U, V, W, u_idx, v_idx, w_idx, start_col):
        # u_idx: list of 4 indices for A block
        # v_idx: list of 4 indices for B block
        # w_idx: list of 4 indices for C block
        
        u_coeffs = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, -1],
            [0, 1, 0, -1]
        ]
        
        v_coeffs = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [1, -1, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        
        w_coeffs = [
            [1, 0, 0, -1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ]
        
        for r in range(7):
            col = start_col + r
            for i in range(4):
                U[u_idx[i], col] += u_coeffs[r][i]
                V[v_idx[i], col] += v_coeffs[r][i]
                W[w_idx[i], col] += w_coeffs[r][i]

    # Block 1
    u1 = [0, 1, 2, 3]
    v1 = [0, 1, 2, 3]
    w1 = [0, 1, 2, 3]
    fill_strassen(U, V, W, u1, v1, w1, 0)
    
    # Block 2
    u2 = [0, 1, 2, 3]
    v2 = [4, 5, 6, 7]
    w2 = [4, 5, 6, 7]
    fill_strassen(U, V, W, u2, v2, w2, 7)
    
    # Block 3
    u3 = [4, 5, 6, 7]
    v3 = [8, 9, 10, 11]
    w3 = [0, 1, 2, 3]
    fill_strassen(U, V, W, u3, v3, w3, 14)
    
    # Block 4
    u4 = [4, 5, 6, 7]
    v4 = [12, 13, 14, 15]
    w4 = [4, 5, 6, 7]
    fill_strassen(U, V, W, u4, v4, w4, 21)
    
    # Part 2: 2x4x1 (Rank 8)
    # A (2x4), B1 (4x1).
    # C1 (2x1).
    # Standard dot products.
    # A indices: 0..7.
    # B1 indices: 16..19 (j=0..3, k=4).
    # C1 indices: 8..9 (k=4, i=0,1).
    
    # c_0 = sum_j A[0,j] B[j,4]
    # c_1 = sum_j A[1,j] B[j,4]
    
    for j in range(4):
        col = 28 + j
        U[j, col] = 1       # A[0, j]
        V[16+j, col] = 1    # B[j, 4]
        W[8, col] = 1       # C[0, 4]
        
    for j in range(4):
        col = 32 + j
        U[4+j, col] = 1     # A[1, j]
        V[16+j, col] = 1    # B[j, 4]
        W[9, col] = 1       # C[1, 4]
        
    # Verify
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank = 40
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
