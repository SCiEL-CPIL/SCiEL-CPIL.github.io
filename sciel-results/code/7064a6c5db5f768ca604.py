import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # 10 seconds safety margin from 150s limit
    
    n, m, p = 2, 3, 6
    
    # Rank 33 solution: 12 (schoolbook for col 0) + 3*7 (Strassen for cols 1-2, 3-4, 5-... wait)
    # Actually: 12 (col 0 schoolbook) + 7*3 (Strassen for 2x2 blocks)
    # Wait, m=3, so A is 2x3. B is 3x6.
    # Split A into A[:,0] (2x1) and A[:,1:3] (2x2).
    # A[:,0] * B[0,:] -> 2x6. Rank = 2*6 = 12.
    # A[:,1:3] * B[1:3,:] -> 2x6.
    # B[1:3,:] is 2x6. Split into three 2x2 blocks? No, 2x6 = three 2x2 blocks.
    # Each 2x2x2 block has rank 7 (Strassen).
    # Total = 12 + 7 + 7 + 7 = 33.
    
    rank = 33
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A[:, 0] (2x1) * B[0, :] (1x6) -> C[:, 0:6]
    # C[i, k] = A[i, 0] * B[0, k]
    for k in range(p):
        for i in range(n):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
    
    # Part 2: A[:, 1:3] (2x2) * B[1:3, :] (2x6)
    # B[1:3, :] has 6 columns. Split into three 2x2 blocks: cols 0-1, 2-3, 4-5.
    # For each block t (0, 1, 2), B[1:3, 2t:2t+2] is 2x2.
    # A[:, 1:3] is 2x2.
    # Use Strassen for each 2x2x2 multiplication.
    
    # Strassen 2x2x2 decomposition:
    # M1 = (a+d)(e+i)
    # M2 = (c+d)e
    # M3 = a(f-i)
    # M4 = d(h-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(i+h)
    # C11 = M5 + M4 - M7 - M1
    # C12 = M3 + M7
    # C21 = M2 + M5
    # C22 = M1 - M2 + M3 - M4
    
    # For our case:
    # A_sub = A[:, 1:3], so A_sub[i, j] where i in {0,1}, j in {0,1} corresponds to A[i, 1+j]
    # B_sub = B[1:3, 2t:2t+2], so B_sub[j, k] where j in {0,1}, k in {0,1} corresponds to B[1+j, 2t+k]
    # C_sub[i, k] corresponds to C[i, 2t+k]
    
    # U indices: i*m + (1+j) = i*3 + 1 + j
    # V indices: (1+j)*p + (2t+k) = (1+j)*6 + 2t + k
    # W indices: (2t+k)*n + i = (2t+k)*2 + i
    
    strassen_terms = [
        # (U_indices, V_indices, W_indices)
        # M1 = (a+d)(e+i)
        ([0, 1], [0, 1], [0, 1]),  # (A00+A11)(B00+B11)
        # M2 = (c+d)e
        ([1, 1], [0], [1]),  # (A10+A11)B00 -> C10 += M2, C11 -= M2
        # M3 = a(f-i)
        ([0], [1, 0], [0, 1]),  # A00(B10-B00) -> C00 -= M3, C01 += M3
        # M4 = d(h-e)
        ([1], [1, 0], [1, 0]),  # A11(B11-B00) -> C00 -= M4, C11 += M4
        # M5 = (a+b)h
        ([0, 1], [1], [0, 1]),  # (A00+A10)B11 -> C00 += M5, C10 += M5
        # M6 = (c-a)(e+f)
        ([1, 0], [0, 1], []),  # (A10-A00)(B00+B10) -> used in C10, C11
        # M7 = (b-d)(i+h)
        ([0, 1], [0, 1], []),  # (A01-A11)(B00+B11) -> used in C00, C01, C11
    ]
    
    # Let me use the standard Strassen formulas more carefully:
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a21 + a22)b11
    # M3 = a11(b21 - b11)
    # M4 = a22(b22 - b11)
    # M5 = (a11 + a12)b22
    # M6 = (a21 - a11)(b11 + b12)
    # M7 = (a12 - a22)(b12 + b22)
    # c11 = M5 + M4 - M7 - M1
    # c12 = M3 + M7
    # c21 = M2 + M5
    # c22 = M1 - M2 + M3 - M4
    
    # Let me re-implement this properly.
    
    for t in range(3):
        # Block t: B[1:3, 2t:2t+2]
        
        # M1 = (A01 + A12)(B10 + B21) where B10=B[1,2t], B21=B[2,2t+1]
        # Wait, let me use 0-based indexing for the sub-blocks.
        # A_sub[i,j] = A[i, 1+j], i,j in {0,1}
        # B_sub[j,k] = B[1+j, 2t+k], j,k in {0,1}
        
        # M1 = (A_sub[0,0] + A_sub[1,1]) * (B_sub[0,0] + B_sub[1,1])
        U[0*3 + 1, col] = 1.0  # A[0,1]
        U[1*3 + 2, col] = 1.0  # A[1,2]
        V[1*6 + 2*t, col] = 1.0    # B[1, 2t]
        V[2*6 + 2*t + 1, col] = 1.0  # B[2, 2t+1]
        W[2*t*2 + 0, col] = -1.0    # C[0, 2t] -= M1
        W[(2*t+1)*2 + 1, col] = 1.0  # C[1, 2t+1] += M1
        col += 1
        
        # M2 = (A_sub[1,0] + A_sub[1,1]) * B_sub[0,0]
        U[1*3 + 1, col] = 1.0  # A[1,1]
        U[1*3 + 2, col] = 1.0  # A[1,2]
        V[1*6 + 2*t, col] = 1.0    # B[1, 2t]
        W[(2*t+1)*2 + 0, col] = 1.0  # C[1, 2t] += M2
        W[(2*t+1)*2 + 1, col] = -1.0  # C[1, 2t+1] -= M2
        col += 1
        
        # M3 = A_sub[0,0] * (B_sub[1,0] - B_sub[0,0])
        U[0*3 + 1, col] = 1.0  # A[0,1]
        V[2*6 + 2*t, col] = 1.0    # B[2, 2t]
        V[1*6 + 2*t, col] = -1.0   # B[1, 2t]
        W[2*t*2 + 0, col] = -1.0    # C[0, 2t] -= M3
        W[(2*t+1)*2 + 0, col] = 1.0  # C[1, 2t] += M3
        col += 1
        
        # M4 = A_sub[1,1] * (B_sub[1,1] - B_sub[0,0])
        U[1*3 + 2, col] = 1.0  # A[1,2]
        V[2*6 + 2*t + 1, col] = 1.0  # B[2, 2t+1]
        V[1*6 + 2*t, col] = -1.0   # B[1, 2t]
        W[2*t*2 + 0, col] = -1.0    # C[0, 2t] -= M4
        W[(2*t+1)*2 + 1, col] = 1.0  # C[1, 2t+1] += M4
        col += 1
        
        # M5 = (A_sub[0,0] + A_sub[0,1]) * B_sub[1,1]
        U[0*3 + 1, col] = 1.0  # A[0,1]
        U[0*3 + 2, col] = 1.0  # A[0,2]
        V[2*6 + 2*t + 1, col] = 1.0  # B[2, 2t+1]
        W[2*t*2 + 0, col] = 1.0    # C[0, 2t] += M5
        W[(2*t+1)*2 + 0, col] = 1.0  # C[1, 2t] += M5
        col += 1
        
        # M6 = (A_sub[1,0] - A_sub[0,0]) * (B_sub[0,0] + B_sub[0,1])
        U[1*3 + 1, col] = 1.0  # A[1,1]
        U[0*3 + 1, col] = -1.0  # A[0,1]
        V[1*6 + 2*t, col] = 1.0    # B[1, 2t]
        V[1*6 + 2*t + 1, col] = 1.0  # B[1, 2t+1]
        W[(2*t+1)*2 + 0, col] = 1.0  # C[1, 2t] += M6
        W[(2*t+1)*2 + 1, col] = 1.0  # C[1, 2t+1] += M6
        col += 1
        
        # M7 = (A_sub[0,1] - A_sub[1,1]) * (B_sub[0,1] + B_sub[1,1])
        U[0*3 + 2, col] = 1.0  # A[0,2]
        U[1*3 + 2, col] = -1.0  # A[1,2]
        V[1*6 + 2*t + 1, col] = 1.0  # B[1, 2t+1]
        V[2*6 + 2*t + 1, col] = 1.0  # B[2, 2t+1]
        W[2*t*2 + 0, col] = -1.0    # C[0, 2t] -= M7
        W[(2*t+1)*2 + 0, col] = 1.0  # C[1, 2t] += M7
        W[(2*t+1)*2 + 1, col] = -1.0  # C[1, 2t+1] -= M7
        col += 1
    
    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, n, m, p):
        # Check that sum_k U[:,k] outer V[:,k] outer W[:,k] = multiplication tensor
        # Multiplication tensor T[i,j,k] = 1 if i is row of A, j is col of A / row of B, k is col of B
        # T[i,j,k] = 1 if A[i,j] * B[j,k] contributes to C[i,k]
        # Actually, T[i,j,k] = 1 for all valid i,j,k in the multiplication
        # The tensor is T[i,j,k] = delta_{j, row_of_B} ... no.
        # Matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
        # Tensor T has T[i,j,k] = 1 for all i in {0..n-1}, j in {0..m-1}, k in {0..p-1}
        # And the decomposition is sum_r U[i,j,r] V[j,k,r] W[k,i,r] = T[i,j,k]
        
        # Let's verify by computing C = A*B for random A, B
        import random
        random.seed(42)
        for _ in range(5):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            # Compute C from decomposition
            C_computed = np.zeros((n, p))
            for r in range(rank):
                A_r = U_mat[:, r].reshape(n, m)
                B_r = V_mat[:, r].reshape(m, p)
                C_r = W_mat[:, r].reshape(p, n)
                # C_computed[i,k] += sum_j A_r[i,j] * B_r[j,k] * C_r[k,i]
                # This is not quite right. Let me think.
                # The decomposition is: sum_r U[:,r] V[:,r] W[:,r] = T
                # Where U is indexed by (i,j), V by (j,k), W by (k,i)
                # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                # For matrix multiplication, T[i,j,k] = 1 for all valid indices
                # So we need: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k
                
            # Alternative verification: check that the tensor matches
            T_computed = np.zeros((n, m, p))
            for r in range(rank):
                U_r = U_mat[:, r].reshape(n, m)
                V_r = V_mat[:, r].reshape(m, p)
                W_r = W_mat[:, r].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_computed[i, j, k] += U_r[i, j] * V_r[j, k] * W_r[k, i]
            
            T_expected = np.ones((n, m, p))
            if not np.allclose(T_computed, T_expected, atol=1e-10):
                return False
        return True
    
    if verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback: schoolbook rank 36
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col_fb] = 1.0
                    V_fb[j * p + k, col_fb] = 1.0
                    W_fb[k * n + i, col_fb] = 1.0
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
