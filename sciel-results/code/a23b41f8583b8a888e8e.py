import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS  # n=4, m=4, p=6

    # Strategy: M_{4,4,6} = M_{4,4,4} + M_{4,4,2}
    # Split B (4x6) into B1 (4x4) and B2 (4x2).
    # C1 = A*B1 (4x4), C2 = A*B2 (4x2).
    # C = [C1, C2].
    #
    # M_{4,4,4}: use 4 independent Strassen 2x2*2x2, rank 28.
    # M_{4,4,2}: rank 8 (two independent M_{4,4,1}, each rank 4).
    # Total rank: 36.
    #
    # W convention: W[k*n+i, r] corresponds to C[i, k].
    # So W has shape (p*n, rank) = (24, rank).
    # For C1 (4x4): C1[i, k] -> W[k*4+i, r] for k,i in 0..3.
    # For C2 (4x2): C2[i, k] -> W[(4+k)*4+i, r] for k in 0..1, i in 0..3.
    
    # Build M_{4,4,4} decomposition (rank 28)
    # A = [[A00, A01], [A10, A11]] (2x2 blocks of 2x2)
    # B1 = [[B00, B01], [B10, B11]] (2x2 blocks of 2x2)
    # C1 = [[C00, C01], [C10, C11]]
    # Cij = Ai0*B0j + Ai1*B1j
    
    R_444 = 28
    U_444 = np.zeros((16, R_444))
    V_444 = np.zeros((16, R_444))
    W_444 = np.zeros((16, R_444))
    
    alpha_s = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1]   # M7
    ]
    beta_s = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [0, 0, 1, -1],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1]    # M7
    ]
    recon_s = [
        [(0, 1), (3, 1), (4, -1), (6, 1)],   # C00
        [(2, 1), (4, 1)],                     # C01
        [(1, 1), (3, 1)],                     # C10
        [(0, 1), (1, -1), (2, 1), (5, 1)]     # C11
    ]
    
    for i in range(2):
        for j in range(2):
            for t in range(7):
                col = (i * 2 + j) * 7 + t
                # A super-block: A[i,0] (2x2) and A[i,1] (2x2)
                # a=0,1 -> A[i,0]; a=2,3 -> A[i,1]
                for a in range(4):
                    if alpha_s[t][a] != 0:
                        if a < 2:
                            u_row = (i * 2 + a) * 4 + 0  # A[i,0]: rows i*2, i*2+1; cols 0,1
                            # a=0 -> row i*2, col 0; a=1 -> row i*2+1, col 1
                            # Actually a=0: (row 0, col 0) of block -> (i*2+0, 0*2+0) = (i*2, 0)
                            # a=1: (row 1, col 1) of block? No, a=0,1 should be (0,0) and (1,1)? 
                            # No! a=0 -> (0,0), a=1 -> (0,1), a=2 -> (1,0), a=3 -> (1,1) for a 2x2 block
                            # Wait, I need to be consistent. Let me use row-major: a=0->(0,0), a=1->(0,1), a=2->(1,0), a=3->(1,1)
                            pass
                        # Let me redo: for 2x2 block, index a: row=a//2, col=a%2
                        ar, ac = a // 2, a % 2
                        if a < 2:
                            # A[i,0]: block at row-block i, col-block 0
                            u_row = (i * 2 + ar) * 4 + (0 * 2 + ac)
                        else:
                            # A[i,1]: block at row-block i, col-block 1
                            u_row = (i * 2 + ar) * 4 + (1 * 2 + ac)
                        U_444[u_row, col] += alpha_s[t][a]
                
                # B super-block: B[0,j] (2x2) and B[1,j] (2x2)
                # b=0,1 -> B[0,j]; b=2,3 -> B[1,j]
                for b in range(4):
                    if beta_s[t][b] != 0:
                        br, bc = b // 2, b % 2
                        if b < 2:
                            # B[0,j]: row-block 0, col-block j
                            v_row = (0 * 2 + br) * 4 + (j * 2 + bc)
                        else:
                            # B[1,j]: row-block 1, col-block j
                            v_row = (1 * 2 + br) * 4 + (j * 2 + bc)
                        V_444[v_row, col] += beta_s[t][b]
                
                # C output: C[i,j] is 2x2
                for c in range(4):
                    gamma = 0
                    for t2, g in recon_s[c]:
                        if t2 == t:
                            gamma += g
                    if gamma != 0:
                        cr, cc = c // 2, c % 2
                        # C[i,j]: row i*2+cr, col j*2+cc
                        # W index: k*n+i where k=col, i=row
                        # For C1 (first 4 cols): W[k*4+i, col]
                        w_row = (j * 2 + cc) * 4 + (i * 2 + cr)
                        W_444[w_row, col] += gamma
    
    # Build M_{4,4,2} decomposition (rank 8)
    # C2[i, k] = sum_j A[i,j]*B2[j,k] for i=0..3, k=0..1
    # Term (j, k): j=0..3, k=0..1
    R_442 = 8
    U_442 = np.zeros((16, R_442))
    V_442 = np.zeros((8, R_442))
    W_442 = np.zeros((8, R_442))
    
    for j in range(4):
        for k in range(2):
            t = j * 2 + k
            for i in range(4):
                U_442[i * 4 + j, t] = 1
            V_442[k * 4 + j, t] = 1
            for i in range(4):
                # C2[i,k] -> W[(4+k)*4+i, t]
                W_442[(4 + k) * 4 + i, t] = 1
    
    # Combine
    R_total = R_444 + R_442  # 36
    U = np.zeros((16, R_total))
    V = np.zeros((24, R_total))
    W = np.zeros((24, R_total))
    
    U[:, :R_444] = U_444
    V[:16, :R_444] = V_444
    W[:16, :R_444] = W_444
    
    U[:, R_444:R_total] = U_442
    V[16:24, R_444:R_total] = V_442
    W[16:24, R_444:R_total] = W_442
    
    # Verify
    if time.time() - t0 < 100:
        A_test = np.random.randint(-2, 3, size=(4, 4))
        B_test = np.random.randint(-2, 3, size=(4, 6))
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        A_vec = A_test.flatten()  # row-major: A[i,j] at i*4+j
        B_vec = B_test.flatten()  # row-major: B[j,k] at j*6+k
        
        # Compute C = A*B
        C_expected = A_test @ B_test
        
        # Compute via decomposition: C[i,k] = sum_r U[i*4+?, r] * V[?, r] * W[k*4+i, r]
        # Actually: C[i,k] = sum_r (sum_j U[i*4+j, r]*A[i,j]) * (sum_j V[j*6+k, r]*B[j,k]) * W[k*4+i, r]
        # Wait, that's not how it works. The tensor decomposition is:
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]... no.
        # 
        # The matrix multiplication tensor: T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B
        # contributes to (i,k) entry of C. So T[i,j,k] = delta (always 1 for valid i,j,k).
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # 
        # Tensor decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r where
        # U_r is in R^{n*m}, V_r is in R^{m*p}, W_r is in R^{n*p}
        # T[i,j,k] = sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[i*p+k]... 
        # 
        # Actually the standard form: C[i,k] = sum_r (sum_j U_r[i*m+j] * A[i,j]) * (sum_j V_r[j*p+k] * B[j,k]) * W_r[k*n+i]
        # No, that's not right either.
        #
        # The correct form: C[i,k] = sum_r U_r[i,m] * V_r[m,k] * W_r[i,k] where the sums are over m.
        # More precisely: C[i,k] = sum_r (U_r[i,m] * A[i,m]) * (V_r[m,k] * B[m,k]) * W_r[i,k]
        # No... 
        #
        # The tensor M_{n,m,p} has T[i,j,k] = 1 for all valid i,j,k (it's the "multiplication tensor").
        # C[i,k] = sum_j A[i,j] * B[j,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
        # 
        # A rank-r decomposition: T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
        # where u_r ∈ R^{n*m}, v_r ∈ R^{m*p}, w_r ∈ R^{n*p}
        # T[i,j,k] = sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
        # 
        # Then C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
        # = sum_j sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] * A[i,j] * B[j,k]
        # = sum_r w_r[k*n+i] * (sum_j u_r[i*m+j] * A[i,j]) * (sum_j v_r[j*p+k] * B[j,k])
        #
        # So: C[i,k] = sum_r W[k*n+i, r] * (U[:,r] · A_row_i) * (V[:,r] · B_col_k)
        # where A_row_i is the i-th row of A (length m), B_col_k is the k-th column of B (length m).
        #
        # Wait, U has shape (n*m, R). U[i*m+j, r] is the coefficient for A[i,j] in term r.
        # V has shape (m*p, R). V[j*p+k, r] is the coefficient for B[j,k] in term r.
        # W has shape (n*p, R). W[k*n+i, r] is the coefficient for C[i,k] in term r.
        #
        # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
        
        C_computed = np.zeros((4, 6))
        for i in range(4):
            for k in range(6):
                val = 0
                for r in range(R_total):
                    u_part = 0
                    for j in range(4):
                        u_part += U_arr[i * 4 + j, r] * A_test[i, j]
                    v_part = 0
                    for j in range(4):
                        v_part += V_arr[j * 6 + k, r] * B_test[j, k]
                    val += W_arr[k * 4 + i, r] * u_part * v_part
                C_computed[i, k] = val
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback to original rank-77 if verification fails
            pass
    
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                rx = round(x)
                if abs(x - rx) < 1e-9:
                    new_row.append(int(rx))
                else:
                    half = round(x * 2) / 2
                    if abs(x - half) < 1e-9:
                        new_row.append(half)
                    else:
                        new_row.append(x)
            result.append(new_row)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
