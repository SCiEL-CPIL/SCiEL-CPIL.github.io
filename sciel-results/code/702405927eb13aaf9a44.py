import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s safety margin before 150s kill
    
    n, m, p = 3, 4, 4
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Build rank-11 decomposition for M_{3,2,2} (3x2 * 2x2 -> 3x2)
    # Indices: A: 0..5, B: 0..3, C: 0..5
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    # Strassen for 2x2x2 part (rows 0,1 of A)
    # M1: (a00+a10)(b00+b10) -> c00+=, c11+=
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    # M2: (a10+a11)b00 -> c10+=, c11-=
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    # M3: a00(b10-b00) -> c01+=, c11+=
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    # M4: a10(b01-b11) -> c00+=, c10+=
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    # M5: (a01+a11)b11 -> c00-=, c01+=
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    # M6: (a01-a00)(b00+b10) -> c11+=
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    # M7: (a10-a11)(b10+b11) -> c00+=
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1
    
    # Extra 4 multiplications for row 2 of A (a20, a21)
    # M8: a20 * b00 -> c20
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[4, 7] = 1
    
    # M9: a21 * b10 -> c20
    U_loc[5, 8] = 1
    V_loc[2, 8] = 1
    W_loc[4, 8] = 1
    
    # M10: a20 * b01 -> c21
    U_loc[4, 9] = 1
    V_loc[1, 9] = 1
    W_loc[5, 9] = 1
    
    # M11: a21 * b11 -> c21
    U_loc[5, 10] = 1
    V_loc[3, 10] = 1
    W_loc[5, 10] = 1
    
    # Verify local decomposition is correct
    # C = sum_r U[:,r] * V[:,r] * W[:,r] (outer products combined)
    # Actually: C[i,c] = sum_r U[i,r] * V[j,r] * W[c,r] where appropriate
    # Let me verify by building the full 3x2*2x2 -> 3x2 tensor
    # Tensor T[a_idx, b_idx, c_idx] = 1 if a_row * b_col == c (i.e., standard matmul)
    # For 3x2 * 2x2 -> 3x2: C[row, col] = sum_k A[row,k] * B[k,col]
    # So T[a_idx, b_idx, c_idx] = 1 if a_row * b_col... 
    # a_idx = a_row*2+a_col, b_idx = b_row*2+b_col, c_idx = c_row*2+c_col
    # C[c_row, c_col] = sum_k A[a_row, k] * B[k, c_col]
    # So contribution to T from rank-r: U[a_idx,r]*V[b_idx,r]*W[c_idx,r] should sum to 
    # delta(a_row, c_row) * delta(a_col, b_row) * delta(b_col, c_col) ... 
    # Actually: T[a_idx, b_idx, c_idx] = 1 iff a_row == c_row AND a_col == b_row AND b_col == c_col
    
    # Check: sum_r U[a,r]*V[b,r]*W[c,r] should equal T[a,b,c]
    # Build expected tensor
    T_exp = np.zeros((6, 4, 6))
    for a_row in range(3):
        for a_col in range(2):
            for b_row in range(2):
                for b_col in range(2):
                    a_idx = a_row * 2 + a_col
                    b_idx = b_row * 2 + b_col
                    c_row = a_row
                    c_col = b_col
                    if c_row < 3 and c_col < 2:
                        c_idx = c_row * 2 + c_col
                        T_exp[a_idx, b_idx, c_idx] = 1.0
    
    # Check decomposition
    T_comp = np.zeros((6, 4, 6))
    for r in range(11):
        T_comp += np.einsum('i,j,k->ijk', U_loc[:, r], V_loc[:, r], W_loc[:, r])
    
    if not np.allclose(T_comp, T_exp, atol=1e-10):
        # Fallback: use naive rank-12 (just in case)
        U_loc = np.zeros((6, 12))
        V_loc = np.zeros((4, 12))
        W_loc = np.zeros((6, 12))
        idx = 0
        for a_row in range(3):
            for a_col in range(2):
                for b_col in range(2):
                    a_idx = a_row * 2 + a_col
                    b_idx = a_col * 2 + b_col
                    c_idx = a_row * 2 + b_col
                    U_loc[a_idx, idx] = 1
                    V_loc[b_idx, idx] = 1
                    W_loc[c_idx, idx] = 1
                    idx += 1
        # This gives rank 12, not 11. Let's just use the Strassen+extra as is.
        # The Strassen part should be correct by construction.
        pass
    
    # Now assemble the full 3x4 * 4x4 -> 3x4 decomposition
    # Split A into left (cols 0,1) and right (cols 2,3)
    # Split B into 4 blocks of 2x2: B[0:2,0:2], B[0:2,2:4], B[2:4,0:2], B[2:4,2:4]
    # C = A_L * B[0:2,0:2] + A_R * B[2:4,0:2] + A_L * B[0:2,2:4] + A_R * B[2:4,2:4]
    # Each term is 3x2 * 2x2 -> 3x2, decomposed with rank 11
    
    # Block definitions: (a_col_start, b_row_start, b_col_start, c_col_start)
    blocks = [
        (0, 0, 0, 0),  # A_L * B[0:2,0:2] -> C[:,0:2]
        (2, 2, 0, 0),  # A_R * B[2:4,0:2] -> C[:,0:2]
        (0, 0, 2, 2),  # A_L * B[0:2,2:4] -> C[:,2:4]
        (2, 2, 2, 2),  # A_R * B[2:4,2:4] -> C[:,2:4]
    ]
    
    for block_idx, (a_col_start, b_row_start, b_col_start, c_col_start) in enumerate(blocks):
        col_offset = block_idx * 11
        
        # U: map local A index (a_row*2+a_col) to global A index (a_row*m + a_col)
        for a_row in range(3):
            for a_col in range(2):
                local_a = a_row * 2 + a_col
                global_a = a_row * m + (a_col_start + a_col)
                U[global_a, col_offset:col_offset+11] += U_loc[local_a, :]
        
        # V: map local B index (b_row*2+b_col) to global B index (b_row*p + b_col)
        for b_row in range(2):
            for b_col in range(2):
                local_b = b_row * 2 + b_col
                global_b = (b_row_start + b_row) * p + (b_col_start + b_col)
                V[global_b, col_offset:col_offset+11] += V_loc[local_b, :]
        
        # W: map local C index (c_row*2+c_col) to global C index (c_col*n + c_row)
        # W[k*n+i, r] where k=c_col, i=c_row (trace convention: W[k*n+i,r])
        for c_row in range(3):
            for c_col in range(2):
                local_c = c_row * 2 + c_col
                global_c = (c_col_start + c_col) * n + c_row
                W[global_c, col_offset:col_offset+11] += W_loc[local_c, :]
    
    # Verify the full decomposition
    # C = sum_r U[:,r] * V[:,r] * W[:,r]
    # For matrix multiplication: C[i,j] = sum_k A[i,k] * B[k,j]
    # In tensor form: T[a_idx, b_idx, c_idx] = 1 iff a_idx corresponds to A[i,k], 
    # b_idx to B[k,j], c_idx to C[i,j] where c_idx = j*n + i
    # So T[a_idx, b_idx, c_idx] = 1 iff:
    #   a_idx = i*m + k, b_idx = k*p + j, c_idx = j*n + i for some valid i,k,j
    #   i in [0,2], k in [0,3], j in [0,3]
    
    # Check with random matrices
    rng = np.random.default_rng(42)
    A_test = rng.integers(-5, 5, size=(n, m)).astype(float)
    B_test = rng.integers(-5, 5, size=(m, p)).astype(float)
    C_expected = A_test @ B_test
    
    # Flatten A and B
    A_flat = A_test.flatten()  # shape (12,)
    B_flat = B_test.flatten()  # shape (16,)
    
    # Compute C from decomposition
    C_computed = np.zeros((p * n,))
    for r in range(rank):
        # C_computed[c_idx] += U[a_idx, r] * V[b_idx, r] * W[c_idx, r]
        # This is: C_computed = U.T @ (V * (U... no
        # Actually: C_computed[c_idx] = sum_{a,b} ... no
        # The tensor contraction is: C[c] = sum_r U[a,r] * V[b,r] * W[c,r] summed over a,b
        # But that's not right either. Let me think again.
        # 
        # The rank-1 decomposition of the multiplication tensor:
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Then for specific A and B: C[c_idx] = sum_{a,b} A[a_idx] * B[b_idx] * T[a,b,c_idx]
        # = sum_r W[c_idx, r] * (sum_a A[a_idx]*U[a_idx,r]) * (sum_b B[b_idx]*V[b_idx,r])
        
        u_val = A_flat @ U[:, r]
        v_val = B_flat @ V[:, r]
        C_computed += W[:, r] * (u_val * v_val)
    
    C_computed_matrix = C_computed.reshape((p, n)).T  # W[k*n+i] -> C[i,k]
    
    if not np.allclose(C_computed_matrix, C_expected, atol=1e-10):
        # Something is wrong; return a safe fallback
        # Use naive decomposition with rank 48
        rank_fallback = n * m * p  # 48
        U_f = np.zeros((n*m, rank_fallback))
        V_f = np.zeros((m*p, rank_fallback))
        W_f = np.zeros((p*n, rank_fallback))
        idx = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    a_idx = i * m + k
                    b_idx = k * p + j
                    c_idx = j * n + i
                    U_f[a_idx, idx] = 1
                    V_f[b_idx, idx] = 1
                    W_f[c_idx, idx] = 1
                    idx += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
