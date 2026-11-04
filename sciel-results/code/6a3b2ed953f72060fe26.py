import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Deadline guard: total time limit is 150s, use 130s as safety margin
    DEADLINE = start_time + 130
    
    # 1. Construct a rank-23 decomposition for 3x3 matrix multiplication.
    # We use a known construction. The standard schoolbook is 27.
    # Strassen's algorithm for 2x2 is rank 7.
    # For 3x3, we can use a decomposition based on the 2x2 Strassen + padding, 
    # or a direct low-rank construction.
    # A well-known rank for 3x3 is 23. 
    # Reference: "The rank of 3x3 matrix multiplication" 
    # One such decomposition can be constructed by embedding the 2x2 Strassen 
    # into a 3x3 context or using specific linear combinations.
    
    # Let's construct a valid rank-23 decomposition for 3x3.
    # If we cannot guarantee a specific 23-rank construction is bug-free quickly,
    # we can fall back to a safe lower bound.
    # However, 23 is the known upper bound. Let's try to generate one.
    
    # Actually, a simpler robust approach:
    # Use Strassen's 7-term algorithm for 2x2, and handle the 3x3 case by 
    # viewing 3x3 as 2x2 with an extra row/col? No, that's not direct.
    
    # Let's use a verified rank-23 decomposition for 3x3.
    # The coefficients for a rank-23 decomposition of 3x3 are non-trivial.
    # To ensure correctness within the time limit, I will implement a 
    # "safe" decomposition that is definitely valid and lower than 216.
    
    # Strategy: 
    # 6x6 = 2x2 blocks of 3x3.
    # If we use Schoolbook 3x3 (rank 27), total rank = 7 * 27 = 189.
    # 189 < 216. This is a guaranteed improvement and easier to verify.
    # Can we do better?
    # If we can find a rank-23 3x3, total = 161.
    # Let's attempt to hardcode a known rank-23 decomposition for 3x3.
    
    # Known Rank 23 Decomposition for 3x3 (from literature, e.g., Strassen or later improvements)
    # Constructing this from scratch is error-prone. 
    # However, we can use the fact that 3x3 can be decomposed using 23 terms.
    # Let's define the 3x3 factors U3, V3, W3 of shape (9, 23).
    
    # I will use a construction based on the "Strassen-like" algorithm for 3x3.
    # Actually, it's safer to use the 189 rank (7 * 27) if I'm unsure of the 23 coefficients.
    # But 189 is still a significant jump from 216.
    
    # Let's try to implement the 7 * 27 = 189 decomposition first as a baseline improvement.
    # Then, if time permits, try to optimize the 3x3 part.
    
    # 1. Schoolbook 3x3 Decomposition (Rank 27)
    def get_3x3_schoolbook():
        n3 = 3
        rank3 = 27
        U3 = np.zeros((9, rank3))
        V3 = np.zeros((9, rank3))
        W3 = np.zeros((9, rank3))
        for i in range(n3):
            for j in range(n3):
                for k in range(n3):
                    col = i*9 + j*3 + k
                    U3[i*3+j, col] = 1.0
                    V3[j*3+k, col] = 1.0
                    W3[k*3+i, col] = 1.0
        return U3, V3, W3

    U3, V3, W3 = get_3x3_schoolbook()
    r3 = 27
    
    # 2. Strassen Formulas for 2x2 block multiplication
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # P1 = (A11 + A22) * (B11 + B22)
    # P2 = (A21 + A22) * B11
    # P3 = A11 * (B12 - B22)
    # P4 = A22 * (B21 - B11)
    # P5 = (A11 + A12) * B22
    # P6 = (A21 - A11) * (B11 + B12)
    # P7 = (A12 - A22) * (B21 + B22)
    
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # We need to map the 3x3 factors to the 6x6 factors.
    # Total rank R = 7 * 27 = 189.
    
    total_rank = 7 * r3
    U6 = np.zeros((36, total_rank))
    V6 = np.zeros((36, total_rank))
    W6 = np.zeros((36, total_rank))
    
    # Helper to add a scaled 3x3 decomposition to the global 6x6 matrices
    # The 3x3 decomposition computes C3 = A3 * B3.
    # U3 has rows indexed by (i, j) in 0..2.
    # V3 has rows indexed by (j, k) in 0..2.
    # W3 has rows indexed by (k, i) in 0..2.
    
    # Global indices for 6x6:
    # A is 6x6. Block A11 is A[0:3, 0:3].
    # Global U row for A[i, j] is i*6 + j.
    # For block (bi, bj) where bi, bj in {0,1} (0=top-left, 1=bottom-right etc):
    # Global row index in U for local (i, j) is (bi*3 + i)*6 + (bj*3 + j).
    
    # Similarly for V (B matrix):
    # B is 6x6. Block B11 is B[0:3, 0:3].
    # Global V row for B[j, k] is j*6 + k.
    # For block (bi, bj) of B (where bi is row block index, bj is col block index):
    # Global row index in V for local (j, k) is (bi*3 + j)*6 + (bj*3 + k).
    
    # For W (C matrix):
    # C is 6x6. Block C11 is C[0:3, 0:3].
    # Global W row for C[k, i] is k*6 + i. (Note: W index is k*n + i, so k is col, i is row).
    # Wait, the problem statement says W[k*n+i, r]. 
    # In my 3x3 construction: W3[k*3+i, col] = 1.
    # So local W index is k*3 + i.
    # Global W index for block (ci, cj) where ci is row block, cj is col block:
    # Local i -> global row ci*3 + i. Local k -> global col cj*3 + k.
    # Global W row index = (cj*3 + k)*6 + (ci*3 + i).
    
    def add_block_product(U_g, V_g, W_g, U3, V3, W3, 
                          a_block, b_block, c_block, 
                          a_sign, b_sign, c_sign, 
                          start_col):
        # a_block: (bi, bj) index of A block in the 2x2 grid
        # b_block: (bi, bj) index of B block
        # c_block: (ci, cj) index of C block (output)
        
        # We are adding terms for P = (A_block * A_sign) * (B_block * B_sign)
        # The result contributes to C_block with weight c_sign.
        # Actually, the Strassen formula is linear.
        # P_k = M_k * N_k.
        # M_k is a linear combo of A blocks. N_k is a linear combo of B blocks.
        # The 3x3 decomposition computes M_k * N_k = sum_r U3[:,r] V3[:,r]^T ...
        # So for each term r in the 3x3 decomposition:
        # U6[global_a_row, start_col+r] += a_sign * U3[local_a_row, r]
        # V6[global_b_row, start_col+r] += b_sign * V3[local_b_row, r]
        # W6[global_c_row, start_col+r] += c_sign * W3[local_c_row, r]
        
        # Note: The "a_sign" applies to the A part. 
        # In Strassen, e.g., P1 = (A11 + A22) * (B11 + B22).
        # This is not a single block multiplication. It's a sum of products?
        # No, Strassen computes products of SUMS.
        # M1 = A11 + A22. N1 = B11 + B22.
        # P1 = M1 * N1.
        # The decomposition of M1 * N1 is not just U3 * V3.
        # M1 is a 3x3 matrix. N1 is a 3x3 matrix.
        # The tensor decomposition of M1 * N1 uses the same U3, V3, W3 structure, 
        # but the inputs are M1 and N1.
        # M1[i, j] = A11[i, j] + A22[i, j].
        # So in the U matrix for P1, the row corresponding to M1[i, j] should have 
        # contributions from both A11[i, j] and A22[i, j].
        
        # Therefore, for a single P_k term in Strassen:
        # We iterate over the 3x3 decomposition terms r (0..26).
        # For each r, we have a basis vector u3_r (length 9), v3_r (length 9), w3_r (length 9).
        # We need to map u3_r to the global U6.
        # u3_r[i, j] corresponds to M_k[i, j].
        # M_k is a linear combination of A blocks.
        # So U6[global_index_of(A_block, i, j), start_col+r] += coeff * u3_r[i, j].
        
        pass # This logic is complex to implement generically in a helper. 
             # Let's hardcode the 7 Strassen steps.

    # Step 3: Explicitly construct the 7 Strassen terms
    
    # P1 = (A11 + A22) * (B11 + B22)
    # A part: A11 (block 0,0) and A22 (block 1,1) with coeff +1.
    # B part: B11 (block 0,0) and B22 (block 1,1) with coeff +1.
    # C part: This product contributes to C11, C12, C21, C22?
    # No, P1 is an intermediate result. It is added to specific C blocks.
    # C11 += P1, C22 += P1.
    # So for the W part, we need to add the W3 contributions to C11 (block 0,0) and C22 (block 1,1).
    
    def add_strassen_term(start_col, 
                          a_blocks, # list of (block_idx, coeff)
                          b_blocks, # list of (block_idx, coeff)
                          c_blocks): # list of (block_idx, coeff)
        
        # a_blocks: e.g., [(0, 1), (1, 1)] for A11 + A22. Block 0 is (0,0), Block 1 is (1,1)?
        # Let's use block indices 0..3 for the 2x2 grid:
        # 0: (0,0) top-left
        # 1: (0,1) top-right
        # 2: (1,0) bottom-left
        # 3: (1,1) bottom-right
        
        for r in range(r3):
            # Deadline check
            if time.time() > DEADLINE:
                # Fallback to schoolbook
                rank = 216
                U_f = np.zeros((36, rank))
                V_f = np.zeros((36, rank))
                W_f = np.zeros((36, rank))
                for i in range(6):
                    for j in range(6):
                        for k in range(6):
                            idx = i*6 + j
                            jk = j*6 + k
                            ki = k*6 + i
                            col = i*36 + j*6 + k
                            U_f[idx, col] = 1.0
                            V_f[jk, col] = 1.0
                            W_f[ki, col] = 1.0
                return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
            
            # Extract the 3x3 basis vectors
            u_vec = U3[:, r]
            v_vec = V3[:, r]
            w_vec = W3[:, r]
            
            # Reshape to 3x3 for easier indexing
            u_mat = u_vec.reshape(3, 3)
            v_mat = v_vec.reshape(3, 3)
            w_mat = w_vec.reshape(3, 3)
            
            # Update U6
            for bi, coeff in a_blocks:
                # bi is block index in 0..3
                # bi=0: (0,0), bi=1: (0,1), bi=2: (1,0), bi=3: (1,1)
                row_block = bi // 2
                col_block = bi % 2
                
                for i in range(3):
                    for j in range(3):
                        val = coeff * u_mat[i, j]
                        if val != 0:
                            global_i = row_block * 3 + i
                            global_j = col_block * 3 + j
                            global_u_idx = global_i * 6 + global_j
                            U6[global_u_idx, start_col + r] += val
            
            # Update V6
            for bi, coeff in b_blocks:
                row_block = bi // 2
                col_block = bi % 2
                
                for j in range(3):
                    for k in range(3):
                        val = coeff * v_mat[j, k]
                        if val != 0:
                            global_j = row_block * 3 + j
                            global_k = col_block * 3 + k
                            global_v_idx = global_j * 6 + global_k
                            V6[global_v_idx, start_col + r] += val
            
            # Update W6
            for bi, coeff in c_blocks:
                # Note: W index is k*n + i.
                # bi is block index in C.
                # C block bi corresponds to row_block, col_block.
                # Local w_mat[k, i] (k is col, i is row in 3x3 local).
                row_block = bi // 2
                col_block = bi % 2
                
                for k in range(3):
                    for i in range(3):
                        val = coeff * w_mat[k, i]
                        if val != 0:
                            global_k = col_block * 3 + k
                            global_i = row_block * 3 + i
                            global_w_idx = global_k * 6 + global_i
                            W6[global_w_idx, start_col + r] += val

    # Strassen Formulas:
    # P1 = (A11 + A22) * (B11 + B22)
    # A11 is block 0, A22 is block 3.
    # B11 is block 0, B22 is block 3.
    # P1 contributes to C11 (block 0) and C22 (block 3).
    add_strassen_term(0, 
                      a_blocks=[(0, 1), (3, 1)], 
                      b_blocks=[(0, 1), (3, 1)], 
                      c_blocks=[(0, 1), (3, 1)])
    
    # P2 = (A21 + A22) * B11
    # A21 is block 2, A22 is block 3.
    # B11 is block 0.
    # P2 contributes to C21 (block 2) and C22 (block 3) with -1?
    # C21 = P2 + P4. C22 = P1 - P2 + P3 + P6.
    # So P2 contributes +1 to C21, -1 to C22.
    add_strassen_term(r3, 
                      a_blocks=[(2, 1), (3, 1)], 
                      b_blocks=[(0, 1)], 
                      c_blocks=[(2, 1), (3, -1)])
    
    # P3 = A11 * (B12 - B22)
    # A11 is block 0.
    # B12 is block 1, B22 is block 3.
    # P3 contributes to C12 (block 1) and C22 (block 3).
    # C12 = P3 + P5. C22 = ... + P3 ...
    # So +1 to C12, +1 to C22.
    add_strassen_term(2*r3, 
                      a_blocks=[(0, 1)], 
                      b_blocks=[(1, 1), (3, -1)], 
                      c_blocks=[(1, 1), (3, 1)])
    
    # P4 = A22 * (B21 - B11)
    # A22 is block 3.
    # B21 is block 2, B11 is block 0.
    # P4 contributes to C11 (block 0) and C21 (block 2).
    # C11 = P1 + P4 - P5 + P7. C21 = P2 + P4.
    # So +1 to C11, +1 to C21.
    add_strassen_term(3*r3, 
                      a_blocks=[(3, 1)], 
                      b_blocks=[(2, 1), (0, -1)], 
                      c_blocks=[(0, 1), (2, 1)])
    
    # P5 = (A11 + A12) * B22
    # A11 is block 0, A12 is block 1.
    # B22 is block 3.
    # P5 contributes to C12 (block 1) and C11 (block 0) with -1.
    # C12 = P3 + P5. C11 = ... - P5 ...
    # So +1 to C12, -1 to C11.
    add_strassen_term(4*r3, 
                      a_blocks=[(0, 1), (1, 1)], 
                      b_blocks=[(3, 1)], 
                      c_blocks=[(1, 1), (0, -1)])
    
    # P6 = (A21 - A11) * (B11 + B12)
    # A21 is block 2, A11 is block 0.
    # B11 is block 0, B12 is block 1.
    # P6 contributes to C22 (block 3).
    # C22 = ... + P6.
    # So +1 to C22.
    add_strassen_term(5*r3, 
                      a_blocks=[(2, 1), (0, -1)], 
                      b_blocks=[(0, 1), (1, 1)], 
                      c_blocks=[(3, 1)])
    
    # P7 = (A12 - A22) * (B21 + B22)
    # A12 is block 1, A22 is block 3.
    # B21 is block 2, B22 is block 3.
    # P7 contributes to C11 (block 0).
    # C11 = ... + P7.
    # So +1 to C11.
    add_strassen_term(6*r3, 
                      a_blocks=[(1, 1), (3, -1)], 
                      b_blocks=[(2, 1), (3, 1)], 
                      c_blocks=[(0, 1)])

    # Deadline guard: if we've used too much time, fall back to schoolbook
    elapsed = time.time() - start_time
    if elapsed > 100:
        rank = 216
        U_f = np.zeros((36, rank))
        V_f = np.zeros((36, rank))
        W_f = np.zeros((36, rank))
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    idx = i*6 + j
                    jk = j*6 + k
                    ki = k*6 + i
                    col = i*36 + j*6 + k
                    U_f[idx, col] = 1.0
                    V_f[jk, col] = 1.0
                    W_f[ki, col] = 1.0
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

    # Fast verification using einsum
    try:
        U3 = U6.reshape(6, 6, -1)
        V3 = V6.reshape(6, 6, -1)
        W3 = W6.reshape(6, 6, -1)
        reconstructed = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        target = np.ones((6, 6, 6))
        if np.allclose(reconstructed, target, atol=1e-6):
            return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
        else:
            # Fallback to schoolbook if Strassen construction failed
            rank = 216
            U_f = np.zeros((36, rank))
            V_f = np.zeros((36, rank))
            W_f = np.zeros((36, rank))
            for i in range(6):
                for j in range(6):
                    for k in range(6):
                        idx = i*6 + j
                        jk = j*6 + k
                        ki = k*6 + i
                        col = i*36 + j*6 + k
                        U_f[idx, col] = 1.0
                        V_f[jk, col] = 1.0
                        W_f[ki, col] = 1.0
            return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    except:
        # Fallback to schoolbook
        rank = 216
        U_f = np.zeros((36, rank))
        V_f = np.zeros((36, rank))
        W_f = np.zeros((36, rank))
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    idx = i*6 + j
                    jk = j*6 + k
                    ki = k*6 + i
                    col = i*36 + j*6 + k
                    U_f[idx, col] = 1.0
                    V_f[jk, col] = 1.0
                    W_f[ki, col] = 1.0
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

# EVOLVE-BLOCK-END
