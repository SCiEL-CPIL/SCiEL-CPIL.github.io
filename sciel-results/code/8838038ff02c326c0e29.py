import numpy as np
import time

def construct():
    n, m, p = 2, 3, 6
    start_time = time.time()
    deadline = start_time + 140
    
    # Rank 22 decomposition based on Strassen for 2x2x2 blocks
    # A is 2x3, B is 3x6
    # Split A into A1 (2x2) and A2 (2x1)
    # Split B into B11 (2x6) and B21 (1x6)
    # C = A1 B11 + A2 B21
    # B11 is 2x6, split into 3 blocks of 2x2: B11_0, B11_1, B11_2
    # C = sum_t A1 B11_t + A2 B21
    
    # Strassen for 2x2x2
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # U (2x2), V (2x2), W (2x2) for each term
    # U flattens to [a, b, c, d] indices 0,1,2,3
    # V flattens to [e, f, g, h] indices 0,1,2,3
    # W flattens to [C00, C10, C01, C11] in col-major k*n+i?
    # W index is k*n + i.
    # C00: k=0, i=0 -> 0
    # C10: k=0, i=1 -> 1
    # C01: k=1, i=0 -> 2
    # C11: k=1, i=1 -> 3
    
    strassen_U = [
        [1, 0, 0, 1], # a+d
        [0, 0, 1, 1], # c+d
        [1, 0, 0, 0], # a
        [0, 0, 0, 1], # d
        [1, 1, 0, 0], # a+b
        [0, 0, 1, -1],# c-a
        [0, 1, 0, -1] # b-d
    ]
    
    strassen_V = [
        [1, 1, 0, 0], # e+h
        [1, 0, 0, 0], # e
        [0, 1, 0, -1],# f-h
        [0, -1, 1, 0] # g-e
        [0, 0, 0, 1], # h
        [1, 1, 0, 0], # e+f
        [0, 0, 1, 1]  # g+h
    ]
    
    # W contributions:
    # C00 = m1 + m4 - m5 + m7
    # C10 = m2 + m4
    # C01 = m3 + m5
    # C11 = m1 - m2 + m3 + m6
    
    strassen_W = [
        [1, 0, 0, 1], # m1 -> C00, C11
        [0, 1, 0, 0], # m2 -> C10
        [0, 0, 1, 0], # m3 -> C01
        [1, 1, 0, 0], # m4 -> C00, C10
        [-1, 0, 1, 0],# m5 -> -C00, C01
        [0, 0, 0, 1], # m6 -> C11
        [1, 0, 0, 0]  # m7 -> C00
    ]
    # Note: W values can be -1, 0, 1.
    
    rank = 22
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Fill Strassen blocks for A1 B11
    # A1 uses A cols 0,1.
    # B11 uses B rows 0,1.
    # B11 is split into 3 blocks of 2 cols.
    
    for t in range(3):
        # B cols for this block: 2t, 2t+1
        for s in range(7):
            r = 7 * t + s
            
            # U part: A[i, j] where i in 0,1; j in 0,1
            for i in range(2):
                for j in range(2):
                    U[i * m + j, r] = strassen_U[s][i * 2 + j]
            
            # V part: B[j, k] where j in 0,1; k in 0,1 (local)
            for j in range(2):
                for k in range(2):
                    global_k = 2 * t + k
                    V[j * p + global_k, r] = strassen_V[s][j * 2 + k]
            
            # W part: C[i, k] where i in 0,1; k in 0,1 (local)
            for i in range(2):
                for k in range(2):
                    global_k = 2 * t + k
                    W[global_k * n + i, r] = strassen_W[s][k * 2 + i]
    
    # Fill A2 B21 part
    # A2 is A col 2.
    # B21 is B row 2.
    r = 21
    
    for i in range(2):
        U[i * m + 2, r] = 1
        
    for k in range(6):
        V[2 * p + k, r] = 1
        
    for i in range(2):
        for k in range(6):
            W[k * n + i, r] = 1
            
    # Verify
    def verify(U_mat, V_mat, W_mat):
        # Check C_ik = sum_j A_ij B_jk
        # Reconstruct C_ik from decomposition
        # C_ik = sum_r U[i,j,r] V[j,k,r] W[k,i,r] ? No.
        # The tensor condition is sum_r U[i,j,r] V[j',k',r] W[k'',i'',r] = delta...
        # For the specific case of matrix multiplication:
        # We want to ensure that the decomposition represents the map.
        # Easiest check: Pick random A, B, compute C = AB, and C' from decomposition.
        
        np.random.seed(42)
        A_test = np.random.randint(-10, 10, (n, m))
        B_test = np.random.randint(-10, 10, (m, p))
        C_test = A_test @ B_test
        
        # Compute C' from decomposition
        # C'_ik = sum_r (sum_j U[i,j,r] A_ij) (sum_k V[j,k,r] B_jk) ... No.
        # The decomposition is:
        # C_ik = sum_r W[k,i,r] * (sum_{j} U[i,j,r] A_ij) * (sum_{j} V[j,k,r] B_jk) ??
        # No, the terms are scalar multiplications m_r.
        # m_r = (sum_{i,j} U[i,j,r] A_ij) * (sum_{j,k} V[j,k,r] B_jk)
        # C_ik = sum_r W[k,i,r] m_r
        
        C_decomp = np.zeros((n, p), dtype=np.int64)
        for r in range(rank):
            u_vec = U_mat[:, r]
            v_vec = V_mat[:, r]
            w_vec = W_mat[:, r]
            
            # Compute scalar product with A
            # u_vec is flattened A.
            prod_A = np.dot(u_vec, A_test.flatten())
            prod_B = np.dot(v_vec, B_test.flatten())
            m_r = prod_A * prod_B
            
            # Add to C
            # w_vec is flattened C (col-major k*n+i)
            C_decomp += np.outer(w_vec.reshape(p, n).T, m_r) # w_vec[k*n+i] -> C[i,k]
            # Wait, w_vec index is k*n + i.
            # w_vec.reshape(p, n) has shape (p, n). Element [k, i] is w_vec[k*n+i].
            # C[i,k] gets contribution w_vec[k*n+i] * m_r.
            # So C_decomp[i,k] += w_vec[k*n+i] * m_r.
            # C_decomp is (n, p).
            # w_vec.reshape(p, n).T is (n, p). Element [i, k] is w_vec[k*n+i].
            # So C_decomp += m_r * w_vec.reshape(p, n).T
            
        if not np.allclose(C_test, C_decomp):
            return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook if verification fails (should not happen)
        schoolbook_rank = n * m * p
        U = np.zeros((n * m, schoolbook_rank), dtype=np.int64)
        V = np.zeros((m * p, schoolbook_rank), dtype=np.int64)
        W = np.zeros((p * n, schoolbook_rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        # Return schoolbook
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
