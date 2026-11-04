import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = 38
    U = np.zeros((n * m, rank))  # (8, 38)
    V = np.zeros((m * p, rank))  # (20, 38)
    W = np.zeros((p * n, rank))  # (10, 38)
    
    col = 0
    
    # For each block pair (A_block, B_block):
    # A_block is 2x2, B_block is 2x5
    # Split B_block into B_22 (2x2) and B_23 (2x3)
    # Use Strassen (rank 7) for 2x2 x 2x2
    # Use schoolbook (rank 12) for 2x2 x 2x3
    
    for block_idx in range(2):
        # A_block columns: block_idx*2 to block_idx*2+1
        # B_block rows: block_idx*2 to block_idx*2+1
        
        # --- Strassen part: 2x2 x 2x2 ---
        # A_block[i,j] for i in 0..1, j in 0..1
        # B_22[i,j] for i in 0..1, j in 0..1 (columns 0,1 of B_block)
        
        # Strassen terms M1..M7
        # Each term is a sum of products of linear combinations
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Output C = A_block * B_22 (2x2)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # We need to decompose each output element as sum of products
        # Each M_k is already a product of two linear forms
        
        # Let's build the tensor decomposition term by term
        
        # Helper: add a term U_vec, V_vec, W_vec
        def add_term(U_vals, V_vals, W_vals):
            nonlocal col
            for i in range(8):
                if U_vals[i] != 0:
                    U[i, col] += U_vals[i]
            for i in range(20):
                if V_vals[i] != 0:
                    V[i, col] += V_vals[i]
            for i in range(10):
                if W_vals[i] != 0:
                    W[i, col] += W_vals[i]
            col += 1
        
        # Indexing:
        # U[i*m + j] = U[i*4 + j] for A[i,j], i in 0..1, j in 0..3
        # V[j*p + k] = V[j*5 + k] for B[j,k], j in 0..3, k in 0..4
        # W[k*n + i] = W[k*2 + i] for C[k,i], k in 0..4, i in 0..1
        
        # A_block[i,j]: U[i*4 + block_idx*2 + j]
        # B_22[i,j]: V[(block_idx*2+i)*5 + j]  (rows block_idx*2..block_idx*2+1, cols 0..1)
        # B_23[i,j]: V[(block_idx*2+i)*5 + 2+j] (rows block_idx*2..block_idx*2+1, cols 2..4)
        # C[k,i]: W[k*2 + i]
        
        # Strassen: 7 terms
        # Each term contributes to output C (2x2) which maps to W indices
        
        # M1 = (A00+A11)(B00+B11)
        # A00 = U[0*4 + block_idx*2 + 0], A11 = U[1*4 + block_idx*2 + 1]
        # B00 = V[(block_idx*2+0)*5 + 0], B11 = V[(block_idx*2+1)*5 + 1]
        # M1 contributes: C00 += 1*M1, C11 += 1*M1
        
        a00 = block_idx*2 + 0  # col in A
        a01 = block_idx*2 + 1  # col in A
        a10 = block_idx*2 + 0  # row 1, col 0 -> index 1*4 + block_idx*2 + 0
        a11 = block_idx*2 + 1  # row 1, col 1 -> index 1*4 + block_idx*2 + 1
        
        # Let me use explicit indices
        # A[i,j] -> U[i*4 + block_idx*2 + j]
        # B[i,j] -> V[(block_idx*2+i)*5 + j] for j in 0..4
        # C[k,i] -> W[k*2 + i]
        
        # Strassen M1: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        U_vals = np.zeros(8)
        V_vals = np.zeros(20)
        W_vals = np.zeros(10)
        U_vals[0*4 + block_idx*2 + 0] = 1  # A[0,0]
        U_vals[1*4 + block_idx*2 + 1] = 1  # A[1,1]
        V_vals[(block_idx*2+0)*5 + 0] = 1  # B[0,0]
        V_vals[(block_idx*2+1)*5 + 1] = 1  # B[1,1]
        # M1 contributes to C[0,0] and C[1,1]
        W_vals[0*2 + 0] = 1  # C[0,0]
        W_vals[1*2 + 1] = 1  # C[1,1]
        add_term(U_vals, V_vals, W_vals)
        
        # M2: (A[1,0]+A[1,1]) * B[0,0]
        U_vals = np.zeros(8)
        V_vals = np.zeros(20)
        W_vals = np.zeros(10)
        U_vals[1*4 + block_idx*2 + 0] = 1
        U_vals[1*4 + block_idx*2 + 1] = 1
        V_vals[(block_idx*2+0)*5 + 0] = 1
        # M2 contributes to C[1,0]
        W_vals[0*2 + 1] = 1  # C[0,1]? Wait, C[i,j] where i is row, j is col
        # C = A*B, C[i,j] = sum_k A[i,k]*B[k,j]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # So M2 contributes to C[1,0] which is W[0*2+1] = W[1]
        W_vals[0*2 + 1] = 1
        add_term(U_vals, V_vals, W_vals)
        
        # M3: A[0,0] * (B[0,1]-B[1,1])
        U_vals = np.zeros(8)
        V_vals = np.zeros(20)
        W_vals = np.zeros(10)
        U_vals[0*4 + block_idx*2 + 0] = 1
        V_vals[(block_idx*2+0)*5 + 1] = 1
        V_vals[(block_idx*2+1)*5 + 1] = -1
        # M3 contributes to C[0,1]
        W_vals[1*2 + 0] = 1  # C[1,0]? No...
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # So M3 contributes to C[0,1] which is W[1*2+0] = W[2]
        W_vals[1*2 + 0] = 1
        add_term(U_vals, V_vals, W_vals)
        
        # M4: A[1,1] * (B[1,0]-B[0,0])
        U_vals = np.zeros(8)
        V_vals = np.zeros(20)
        W_vals = np.zeros(10)
        U_vals[1*4 + block_idx*2 + 1] = 1
        V_vals[(block_idx*2+1)*5 + 0] = 1
        V_vals[(block_idx*2+0)*5 + 0] = -1
        # M4 contributes to C[0,0] and C[1,0]
        W_vals[0*2 + 0] = 1  # C[0,0]
        W_vals[0*2 + 1] = 1  # C[1,0]
        add_term(U_vals, V_vals, W_vals)
        
        # M5: (A[0,0]+A[0,1]) * B[1,1]
        U_vals = np.zeros(8)
        V_vals = np.zeros(20)
        W_vals = np.zeros(10)
        U_vals[0*4 + block_idx*2 + 0] = 1
        U_vals[0*4 + block_idx*2 + 1] = 1
        V_vals[(block_idx*2+1)*5 + 1] = 1
        # M5 contributes to C[0,1] with -1, and C[0,0] with -1
        # From above: C00 = M1+M4-M5+M7, C01 = M3+M5
        W_vals[0*2 + 0] = -1  # C[0,0]
        W_vals[1*2 + 0] = 1   # C[0,1]
        add_term(U_vals, V_vals, W_vals)
        
        # M6: (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
        U_vals = np.zeros(8)
        V_vals = np.zeros(20)
        W_vals = np.zeros(10)
        U_vals[1*4 + block_idx*2 + 0] = 1
        U_vals[0*4 + block_idx*2 + 0] = -1
        V_vals[(block_idx*2+0)*5 + 0] = 1
        V_vals[(block_idx*2+0)*5 + 1] = 1
        # M6 contributes to C[1,1]
        W_vals[1*2 + 1] = 1  # C[1,1]
        add_term(U_vals, V_vals, W_vals)
        
        # M7: (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
        U_vals = np.zeros(8)
        V_vals = np.zeros(20)
        W_vals = np.zeros(10)
        U_vals[0*4 + block_idx*2 + 1] = 1
        U_vals[1*4 + block_idx*2 + 1] = -1
        V_vals[(block_idx*2+1)*5 + 0] = 1
        V_vals[(block_idx*2+1)*5 + 1] = 1
        # M7 contributes to C[0,0]
        W_vals[0*2 + 0] = 1  # C[0,0]
        add_term(U_vals, V_vals, W_vals)
        
        # --- Schoolbook part: 2x2 x 2x3 ---
        # C[i,j] for j in 2..4: C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
        # 12 terms: for i in 0..1, j in 2..4, k in 0..1
        for i in range(2):
            for j in range(2, 5):
                for k in range(2):
                    U_vals = np.zeros(8)
                    V_vals = np.zeros(20)
                    W_vals = np.zeros(10)
                    U_vals[i*4 + block_idx*2 + k] = 1
                    V_vals[(block_idx*2+k)*5 + j] = 1
                    W_vals[j*2 + i] = 1  # C[i,j] -> W[j*2+i]
                    add_term(U_vals, V_vals, W_vals)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
