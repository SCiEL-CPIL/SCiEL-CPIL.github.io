import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 120.0

    # Strassen 2x2x2 (Rank 7)
    # Indices for A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # U2[i, t] is coeff for A[i] in term t
    # V2[j, t] is coeff for B[j] in term t
    # W2[k, t] is coeff for C[k] in term t
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ])
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ])
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 1, 0, 1, 0, 0, 0],   # C01
        [0, 0, 1, 0, 1, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ])
    
    # Build 4x4x4 (Rank 49)
    # A is 4x4, B is 4x4, C is 4x4
    # Indices: A[i,j] -> i*4+j, B[j,k] -> j*4+k, C[k,i] -> k*4+i
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Strassen for 4x4x4 uses 7 products of 2x2x2 matrices
    # The 7 products correspond to the 7 terms in Strassen 2x2x2 applied to blocks of 2x2
    # Let's map the 49 terms: 7 block-products * 7 internal terms
    
    # Block indices for 4x4 matrix split into 2x2 blocks
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # Same for B and C.
    
    # The 7 Strassen terms for the 4x4x4 problem (acting on 2x2 blocks):
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # We need to express U4, V4, W4 based on these.
    # For each of the 7 block-products (let's call them P1..P7), we apply the 2x2x2 Strassen decomposition.
    # So for Pq (q=0..6), we have 7 internal terms (t=0..6).
    # Total 49 terms.
    
    # Let's define the coefficients for A, B, C blocks for each Pq.
    # A_blocks: 00, 01, 10, 11
    # B_blocks: 00, 01, 10, 11
    
    # P1: A: 00(+1), 11(+1); B: 00(+1), 11(+1)
    # P2: A: 10(+1), 11(+1); B: 00(+1)
    # P3: A: 00(+1); B: 01(+1), 11(-1)
    # P4: A: 11(+1); B: 10(+1), 00(-1)
    # P5: A: 00(+1), 01(+1); B: 11(+1)
    # P6: A: 10(+1), 00(-1); B: 00(+1), 01(+1)
    # P7: A: 01(+1), 11(-1); B: 10(+1), 11(+1)
    
    # C_blocks for each Pq (how Pq contributes to C):
    # C00 = P1 + P4 - P5 + P7  (Wait, standard Strassen: C00 = M1 + M4 - M5 + M7? No.)
    # Standard Strassen 2x2x2:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's verify indices.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A*B = [[C00, C01], [C10, C11]]
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's code the mapping.
    
    # For each block product q (0..6), and each internal term t (0..6):
    # Term index in 49: q*7 + t
    
    # U4[row, term] += coeff_A_block * U2[local_row, t]
    # V4[col, term] += coeff_B_block * V2[local_col, t]
    # W4[out, term] += coeff_C_block * W2[local_out, t]
    
    # We need to map global indices to local indices.
    # Global A index: i*4+j. Block (bi, bj). Local (i, j) in block.
    # i = bi*2 + li, j = bj*2 + lj.
    # Global index = (bi*2+li)*4 + (bj*2+lj)
    
    # Let's define the block coefficients for A, B, C for each q.
    # Blocks: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    A_block_coeffs = [
        {0: 1, 3: 1},   # q=0 (M1)
        {2: 1, 3: 1},   # q=1 (M2)
        {0: 1},         # q=2 (M3)
        {3: 1},         # q=3 (M4)
        {0: 1, 1: 1},   # q=4 (M5)
        {2: 1, 0: -1},  # q=5 (M6)
        {1: 1, 3: -1}   # q=6 (M7)
    ]
    
    B_block_coeffs = [
        {0: 1, 3: 1},   # q=0 (M1)
        {0: 1},         # q=1 (M2)
        {1: 1, 3: -1},  # q=2 (M3)
        {2: 1, 0: -1},  # q=3 (M4)
        {3: 1},         # q=4 (M5)
        {0: 1, 1: 1},   # q=5 (M6)
        {2: 1, 3: 1}    # q=6 (M7)
    ]
    
    C_block_coeffs = [
        {0: 1, 3: 1},   # q=0 (M1) -> C00, C11
        {2: 1, 3: -1},  # q=1 (M2) -> C10, -C11
        {1: 1, 3: 1},   # q=2 (M3) -> C01, C11
        {0: 1, 2: 1},   # q=3 (M4) -> C00, C10
        {0: -1, 1: 1},  # q=4 (M5) -> -C00, C01
        {3: 1},         # q=5 (M6) -> C11
        {0: 1}          # q=6 (M7) -> C00
    ]
    
    # Fill U4
    for q in range(7):
        for t in range(7):
            term = q*7 + t
            # A part
            for block_idx, coeff in A_block_coeffs[q].items():
                bi = block_idx // 2
                bj = block_idx % 2
                for li in range(2):
                    for lj in range(2):
                        global_i = bi*2 + li
                        global_j = bj*2 + lj
                        global_idx = global_i*4 + global_j
                        U4[global_idx, term] += coeff * U2[li*2+lj, t]
            
            # B part
            for block_idx, coeff in B_block_coeffs[q].items():
                bi = block_idx // 2
                bj = block_idx % 2
                for li in range(2):
                    for lj in range(2):
                        global_j = bi*2 + li
                        global_k = bj*2 + lj
                        global_idx = global_j*4 + global_k
                        V4[global_idx, term] += coeff * V2[li*2+lj, t]
            
            # C part
            for block_idx, coeff in C_block_coeffs[q].items():
                bi = block_idx // 2
                bj = block_idx % 2
                for li in range(2):
                    for lj in range(2):
                        global_k = bi*2 + li
                        global_i = bj*2 + lj
                        global_idx = global_k*4 + global_i
                        W4[global_idx, term] += coeff * W2[li*2+lj, t]
    
    # Build 4x4x8 (Rank 98)
    # A is 4x4 (16 elements)
    # B is 4x8 (32 elements)
    # C is 8x4 (32 elements)
    
    # We treat B as two 4x4 matrices: B_left (cols 0-3) and B_right (cols 4-7).
    # C is split into C_left (cols 0-3 of output? No, C is 8x4. Rows 0-3 and 4-7?)
    # Wait, C[k, i]. k is 0..7. i is 0..3.
    # If we split B into B1 (cols 0-3) and B2 (cols 4-7), then C is split into C1 (rows 0-3) and C2 (rows 4-7).
    # C1 = A * B1
    # C2 = A * B2
    
    # So we can use the 4x4x4 decomposition twice.
    # Term 0..48: for B1 and C1.
    # Term 49..97: for B2 and C2.
    
    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))
    
    # U is same for both halves (A is the same)
    U[:, :49] = U4
    U[:, 49:] = U4
    
    # V
    # B1 is rows 0-3, cols 0-3 of B. Global B index: j*8 + k.
    # For B1, k is 0..3. So global index j*8 + k.
    # V4[j*4+k, term] corresponds to B1[j, k].
    # So V[j*8+k, term] = V4[j*4+k, term] for term 0..48.
    
    for j in range(4):
        for k in range(4):
            V[j*8+k, :49] = V4[j*4+k, :49]
            V[j*8+k+4, 49:] = V4[j*4+k, 49:] # B2 has same A, so same V4 structure?
            # Wait, B2 is cols 4-7.
            # Global index for B2[j, k] is j*8 + (k+4).
            # The multiplication is A * B2.
            # The tensor structure is identical to A * B1.
            # So yes, V[j*8+k+4, 49+t] = V4[j*4+k, t].
    
    # W
    # C1 is rows 0-3 of C. Global C index: k*4 + i.
    # W4[k*4+i, term] corresponds to C1[k, i].
    # So W[k*4+i, term] = W4[k*4+i, term] for term 0..48.
    
    # C2 is rows 4-7 of C. Global C index: (k+4)*4 + i.
    # W[(k+4)*4+i, 49+t] = W4[k*4+i, t].
    
    for k in range(4):
        for i in range(4):
            W[k*4+i, :49] = W4[k*4+i, :49]
            W[(k+4)*4+i, 49:] = W4[k*4+i, 49:]
    
    # Verification (optional, but good for debugging)
    # Let's do a quick check with random matrices if time permits
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, (4, 4)).astype(float)
        B_test = np.random.randint(-5, 5, (4, 8)).astype(float)
        
        # Calculate expected C
        C_test = A_test @ B_test
        
        # Calculate C from decomposition
        # C = W @ diag(S) @ (U^T A) (V^T B)?
        # Standard decomposition: C_{ki} = sum_t W_{ki, t} * S_t
        # S_t = sum_{ij} U_{ij, t} A_{ij} V_{jk, t} ... wait.
        # Tensor: T_{ijk} = sum_t U_{ij, t} V_{jk, t} W_{ki, t}
        # Here A is 4x4, B is 4x8, C is 8x4.
        # Indices: A[i,j], B[j,k], C[k,i].
        # U shape (16, 98) -> U[i*4+j, t]
        # V shape (32, 98) -> V[j*8+k, t]
        # W shape (32, 98) -> W[k*4+i, t]
        
        # S_t = sum_{i,j} A[i,j] U[i*4+j, t] * sum_{j,k} B[j,k] V[j*8+k, t]? No.
        # It's a trilinear form.
        # S_t = sum_{i,j,k} A[i,j] B[j,k] U[i*4+j, t] V[j*8+k, t]
        # Wait, U depends on i,j. V depends on j,k.
        # So S_t = sum_{i,j,k} A[i,j] B[j,k] U[i*4+j, t] V[j*8+k, t]
        
        # Let's compute S for each t
        S = np.zeros(98)
        for t in range(98):
            # sum over i, j, k
            # We can vectorize this
            # A_flat = A.reshape(16)
            # B_flat = B.reshape(32)
            # U_t = U[:, t] (16)
            # V_t = V[:, t] (32)
            # This is not a simple dot product because U and V share the index j.
            
            # Let's do it loop-based for clarity in verification
            s_val = 0.0
            for i in range(4):
                for j in range(4):
                    u_val = A_test[i, j] * U[i*4+j, t]
                    if u_val != 0:
                        for k in range(8):
                            s_val += u_val * B_test[j, k] * V[j*8+k, t]
            S[t] = s_val
        
        # Reconstruct C
        C_recon = np.zeros((8, 4))
        for k in range(8):
            for i in range(4):
                c_val = 0.0
                for t in range(98):
                    c_val += S[t] * W[k*4+i, t]
                C_recon[k, i] = c_val
        
        # Check
        if not np.allclose(C_test, C_recon, atol=1e-5):
            print("Verification failed!")
            # Fallback to schoolbook?
            # Schoolbook rank 128
            U_school = np.zeros((16, 128))
            V_school = np.zeros((32, 128))
            W_school = np.zeros((32, 128))
            
            for i in range(4):
                for j in range(4):
                    for k in range(8):
                        t = (i*4 + j)*8 + k
                        U_school[i*4+j, t] = 1
                        V_school[j*8+k, t] = 1
                        W_school[k*4+i, t] = 1
            
            return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
