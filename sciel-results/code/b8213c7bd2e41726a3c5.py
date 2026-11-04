import numpy as np
import time

def construct():
    n, m, p = 2, 3, 6
    start_time = time.time()
    deadline = start_time + 145
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-8):
        rank = U.shape[1]
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def schoolbook_decomp(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    def strassen_2x2x2_block(A_rows, B_cols, C_rows, C_cols, U, V, W, col):
        """Apply Strassen decomposition for 2x2x2 block.
        A_rows: [r0, r1] rows of A
        B_cols: [c0, c1] cols of B  
        C_rows: [cr0, cr1] rows of C
        C_cols: [cc0, cc1] cols of C
        Returns new col index
        """
        # M1 = (A00+A11)(B00+B11)
        U[A_rows[0], col] += 1.0; U[A_rows[1], col] += 1.0
        V[B_cols[0], col] += 1.0; V[B_cols[1], col] += 1.0
        W[C_rows[0], col] += 1.0; W[C_rows[1], col] += 1.0
        col += 1
        
        # M2 = (A10+A11)B00
        U[A_rows[1], col] += 1.0  # A10 is row1,col0 -> but A is 2x2, A[1,0]
        # Wait, let me be more careful with indices
        # For 2x2x2: A[i,j], B[j,k], C[i,k]
        # A_rows = [i0, i1], B_cols = [j0, j1], C_rows = [i0, i1], C_cols = [j0, j1]
        # Actually this is getting confusing. Let me use the standard Strassen.
        # Let me redo this properly.
        col -= 1  # undo M1 for now, will redo properly
        
        # Standard Strassen for 2x2x2:
        # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]], C = A*B
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b01-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        # c00 = M1+M4-M5+M7
        # c01 = M3+M5
        # c10 = M2+M4
        # c11 = M1-M2+M3+M6
        
        # In our tensor notation:
        # U[i*m+j, r] = coefficient for A[i,j]
        # V[j*p+k, r] = coefficient for B[j,k]  
        # W[k*n+i, r] = coefficient for C[k,i]
        
        i0, i1 = A_rows  # rows of A: 0, 1
        j0, j1 = B_cols   # cols of B (which are rows of B in the 2x2 sense): 0, 1
        # For the 2x2x2 subproblem, we have:
        # A_sub[i, j] where i in {0,1}, j in {0,1}
        # B_sub[j, k] where j in {0,1}, k in {0,1}
        # C_sub[i, k] where i in {0,1}, k in {0,1}
        
        # In global indices:
        # A[i,j] -> U[i*m + j, r]
        # B[j,k] -> V[j*p + k, r]
        # C[i,k] -> W[k*n + i, r]
        
        # M1 = (a00+a11)(b00+b11)
        U[i0 * m + j0, col] += 1.0  # a00
        U[i1 * m + j1, col] += 1.0  # a11
        V[j0 * p + j0, col] += 1.0  # b00 (j=0, k=0)
        V[j1 * p + j1, col] += 1.0  # b11 (j=1, k=1)
        W[j0 * n + i0, col] += 1.0  # c00
        W[j1 * n + i1, col] += 1.0  # c11
        col += 1
        
        # M2 = (a10+a11)b00
        U[i1 * m + j0, col] += 1.0  # a10
        U[i1 * m + j1, col] += 1.0  # a11
        V[j0 * p + j0, col] += 1.0  # b00
        W[j0 * n + i1, col] += 1.0  # c10
        W[j1 * n + i1, col] += -1.0 # c11
        col += 1
        
        # M3 = a00(b10-b11)
        U[i0 * m + j0, col] += 1.0  # a00
        V[j1 * p + j0, col] += 1.0  # b10 (j=1, k=0)
        V[j1 * p + j1, col] += -1.0 # b11
        W[j0 * n + i0, col] += 1.0  # c00
        W[j1 * n + i0, col] += 1.0  # c01
        col += 1
        
        # M4 = a11(b01-b00)
        U[i1 * m + j1, col] += 1.0  # a11
        V[j0 * p + j1, col] += 1.0  # b01 (j=0, k=1)
        V[j0 * p + j0, col] += -1.0 # b00
        W[j0 * n + i0, col] += 1.0  # c00
        W[j1 * n + i0, col] += 1.0  # c01
        col += 1
        
        # M5 = (a00+a01)b11
        U[i0 * m + j0, col] += 1.0  # a00
        U[i0 * m + j1, col] += 1.0  # a01
        V[j1 * p + j1, col] += 1.0  # b11
        W[j0 * n + i0, col] += -1.0 # c00
        W[j1 * n + i0, col] += 1.0  # c01
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[i1 * m + j0, col] += 1.0  # a10
        U[i0 * m + j0, col] += -1.0 # a00
        V[j0 * p + j0, col] += 1.0  # b00
        V[j0 * p + j1, col] += 1.0  # b01
        W[j1 * n + i1, col] += 1.0  # c11
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[i0 * m + j1, col] += 1.0  # a01
        U[i1 * m + j1, col] += -1.0 # a11
        V[j1 * p + j0, col] += 1.0  # b10
        V[j1 * p + j1, col] += 1.0  # b11
        W[j0 * n + i0, col] += 1.0  # c00
        col += 1
        
        return col

    # Build rank-33 decomposition
    total_rank = 33
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # Part 1: A[:,0] * B[0,:] (2x1 * 1x6) -> Rank 12
    for k in range(p):
        for i in range(n):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
    
    # Part 2: A[:,1:3] * B[1:3,:] (2x2 * 2x6) -> 3 * Strassen(2x2x2) -> Rank 21
    for t in range(3):
        k0 = 2 * t
        k1 = 2 * t + 1
        col = strassen_2x2x2_block([0, 1], [1, 2], [0, 1], [k0, k1], U, V, W, col)
    
    if verify_decomposition(U, V, W, n, m, p):
        best_rank = 33
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    else:
        # Fallback to schoolbook
        U, V, W = schoolbook_decomp(n, m, p)
        best_rank = 36
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to search for lower rank
    if time.time() < deadline - 30:
        # Try rank 32
        for target_rank in [32]:
            if time.time() > deadline - 15:
                break
            
            # Random search with refinement
            np.random.seed(42)
            found = False
            
            for trial in range(50):
                if time.time() > deadline - 15:
                    break
                
                U_try = np.random.randn(n * m, target_rank)
                V_try = np.random.randn(m * p, target_rank)
                W_try = np.random.randn(p * n, target_rank)
                
                # Simple iterative refinement
                for it in range(200):
                    if time.time() > deadline - 15:
                        break
                    
                    # Update W
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                Uij = U_try[i * m + j, :]
                                Vjk = V_try[j * p + k, :]
                                prod = Uij * Vjk
                                norm = np.sum(prod * prod)
                                if norm > 1e-15:
                                    W_try[k * n + i, :] = (prod + 0.1 * W_try[k * n + i, :]) / (1.1 * norm) * np.sum(prod * W_try[k * n + i, :])
                    
                    # Update U
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                Vjk = V_try[j * p + k, :]
                                Wki = W_try[k * n + i, :]
                                prod = Vjk * Wki
                                norm = np.sum(prod * prod)
                                if norm > 1e-15:
                                    U_try[i * m + j, :] = (prod + 0.1 * U_try[i * m + j, :]) / (1.1 * norm) * np.sum(prod * U_try[i * m + j, :])
                    
                    # Update V
                    for j in range(m):
                        for k in range(p):
                            for i in range(n):
                                Uij = U_try[i * m + j, :]
                                Wki = W_try[k * n + i, :]
                                prod = Uij * Wki
                                norm = np.sum(prod * prod)
                                if norm > 1e-15:
                                    V_try[j * p + k, :] = (prod + 0.1 * V_try[j * p + k, :]) / (1.1 * norm) * np.sum(prod * V_try[j * p + k, :])
                
                if verify_decomposition(U_try, V_try, W_try, n, m, p, tol=1e-6):
                    found = True
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                    break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

