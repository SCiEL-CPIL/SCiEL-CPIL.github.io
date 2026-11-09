import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # We will construct a rank-110 decomposition for 5x5 matrix multiplication.
    # Strategy: Partition 5x5 matrices into 4x4 and 1x1 blocks.
    # A = [[A44, A41], [A14, A11]] where A44 is 4x4, A41 is 4x1, A14 is 1x4, A11 is 1x1.
    # B = [[B44, B41], [B14, B11]]
    # C = A B = [[C44, C41], [C14, C11]]
    #
    # C44 = A44 B44 + A41 B14
    # C41 = A44 B41 + A41 B11
    # C14 = A14 B44 + A11 B14
    # C11 = A14 B41 + A11 B11
    #
    # We use Strassen (rank 49) for the 4x4 * 4x4 multiplication (A44 B44).
    # We use Schoolbook (rank 16) for the 4x1 * 1x4, 4x4 * 4x1, 1x4 * 4x4 multiplications.
    # We use Schoolbook (rank 4) for the 4x1 * 1x1, 1x1 * 1x4, 1x4 * 4x1 multiplications.
    # We use Schoolbook (rank 1) for the 1x1 * 1x1 multiplication.
    #
    # Total Rank = 49 (A44 B44) 
    #             + 16 (A41 B14: 4*1*4) 
    #             + 16 (A44 B41: 4*4*1) 
    #             + 4  (A41 B11: 4*1*1) 
    #             + 16 (A14 B44: 1*4*4) 
    #             + 4  (A11 B14: 1*1*4) 
    #             + 4  (A14 B41: 1*4*1) 
    #             + 1  (A11 B11: 1*1*1)
    #             = 110.
    
    # Helper to create schoolbook factors for a specific sub-block multiplication
    # n_sub, m_sub, p_sub are dimensions of the sub-matrices
    # offset_i, offset_j are the starting row/col indices in the full 5x5 matrices for A
    # offset_k, offset_l are the starting row/col indices in the full 5x5 matrices for B
    # offset_c_i, offset_c_k are the starting row/col indices in the full 5x5 matrices for C
    # rank_offset is where to start writing in the U, V, W matrices
    
    def add_schoolblock(n_sub, m_sub, p_sub, off_i, off_j, off_k, off_l, off_ci, off_ck, rank_offset, U, V, W):
        rank = n_sub * m_sub * p_sub
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    r = rank_offset + i * m_sub * p_sub + j * p_sub + k
                    
                    # U index: (i_global) * m + (j_global)
                    u_idx = (off_i + i) * m + (off_j + j)
                    # V index: (k_global) * p + (l_global) -- Wait, V is (m*p, R). 
                    # V indices correspond to B[j, k]. 
                    # In the full matrix, B's row index is off_k + k (local k), col is off_l + l (local l).
                    # So V index is (off_k + k) * p + (off_l + l).
                    v_idx = (off_k + k) * p + (off_l + l)
                    
                    # W index: (k_global) * n + (i_global) -- W is (p*n, R).
                    # W indices correspond to C[i, k].
                    # In the full matrix, C's row index is off_ci + i, col is off_ck + k.
                    # Wait, the task says W[k*n+i, r]. 
                    # So W index is (col_global) * n + (row_global).
                    # C's row is off_ci + i. C's col is off_ck + k.
                    w_idx = (off_ck + k) * n + (off_ci + i)
                    
                    U[u_idx, r] = 1
                    V[v_idx, r] = 1
                    W[w_idx, r] = 1
        return rank

    # Helper to create Strassen 4x4 factors
    # We will hardcode the 49 terms for 4x4 Strassen.
    # This is the most error-prone part. 
    # Instead of hardcoding 49 terms manually which is prone to typos, 
    # we can generate them from the 2x2 Strassen logic recursively or use a known explicit list.
    # Let's try to generate the 4x4 Strassen factors by composing 2x2 Strassen factors.
    
    def get_strassen_4x4_factors():
        # Returns U_44 (16, 49), V_44 (16, 49), W_44 (16, 49)
        # Indices for 4x4:
        # A: 0..15 (row*4+col)
        # B: 0..15 (row*4+col)
        # C: 0..15 (row*4+col) -- but W is (p*n) so (col*4+row)
        
        U_44 = np.zeros((16, 49))
        V_44 = np.zeros((16, 49))
        W_44 = np.zeros((16, 49))
        
        # 2x2 Strassen factors
        # Local indices 0..3 for 2x2 matrices.
        # U_22 (4, 7), V_22 (4, 7), W_22 (4, 7)
        
        # Let's define the 2x2 Strassen terms explicitly.
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A11)(B01+B11)
        # M7 = (A01-A11)(B00+B10)
        
        # We will store the contributions for each of the 7 products.
        # Each product is a 2x2 matrix multiplication.
        # So for each product r in 0..6, we need to add 7 terms (one for each scalar mult in the 2x2 Strassen).
        # Total 49 terms.
        
        # Let's define the 2x2 Strassen coefficients.
        # U_22[i, s] where i is A index (0..3), s is term (0..6)
        # V_22[j, s] where j is B index (0..3), s is term (0..6)
        # W_22[k, s] where k is C index (0..3, but W is col*2+row?), s is term (0..6)
        
        # Note: The 2x2 Strassen algorithm computes C = A B.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's create the 2x2 factors.
        U_22 = np.zeros((4, 7))
        V_22 = np.zeros((4, 7))
        W_22 = np.zeros((4, 7)) # W index is col*2+row. C is 2x2.
        
        # Term 0 (M1): (A00+A11)(B00+B11)
        # A: 00, 11. B: 00, 11. C: 00, 11.
        U_22[0, 0] = 1; U_22[3, 0] = 1
        V_22[0, 0] = 1; V_22[3, 0] = 1
        # C00 (idx 0), C11 (idx 3 in W? W[1,1] -> 1*2+1=3)
        W_22[0, 0] = 1; W_22[3, 0] = 1
        
        # Term 1 (M2): (A10+A11)B00
        # A: 10 (idx 2), 11 (idx 3). B: 00 (idx 0). C: 10 (idx 2 in W? W[0,1] -> 0*2+1=2? No. C10 is row 1, col 0. W index = col*2+row = 0*2+1=1? Wait.
        # C[i,k]. W[k*n+i]. For 2x2, n=2. W[k*2+i].
        # C10: i=1, k=0. W[0*2+1] = W[1].
        # C11: i=1, k=1. W[1*2+1] = W[3].
        # C00: i=0, k=0. W[0].
        # C01: i=0, k=1. W[1*2+0] = W[2].
        
        # Let's re-verify W indices for 2x2.
        # C = [[C00, C01], [C10, C11]]
        # W is (p*n) = (2*2) = 4.
        # W[k*n+i].
        # C00 (i=0, k=0) -> W[0]
        # C01 (i=0, k=1) -> W[2]
        # C10 (i=1, k=0) -> W[1]
        # C11 (i=1, k=1) -> W[3]
        
        # Term 1 (M2): (A10+A11)B00 -> C10
        U_22[2, 1] = 1; U_22[3, 1] = 1
        V_22[0, 1] = 1
        W_22[1, 1] = 1 # C10
        
        # Term 2 (M3): A00(B01-B11) -> C01, C11
        U_22[0, 2] = 1
        V_22[1, 2] = 1; V_22[3, 2] = -1
        W_22[2, 2] = 1 # C01
        W_22[3, 2] = 1 # C11
        
        # Term 3 (M4): A11(B10-B00) -> C00, C10
        U_22[3, 3] = 1
        V_22[2, 3] = 1; V_22[0, 3] = -1
        W_22[0, 3] = 1 # C00
        W_22[1, 3] = 1 # C10
        
        # Term 4 (M5): (A00+A01)B11 -> C00, C01
        U_22[0, 4] = 1; U_22[1, 4] = 1
        V_22[3, 4] = 1
        W_22[0, 4] = -1 # C00
        W_22[2, 4] = 1  # C01
        
        # Term 5 (M6): (A10-A11)(B01+B11) -> C11
        U_22[2, 5] = 1; U_22[3, 5] = -1
        V_22[1, 5] = 1; V_22[3, 5] = 1
        W_22[3, 5] = 1  # C11
        
        # Term 6 (M7): (A01-A11)(B00+B10) -> C00
        U_22[1, 6] = 1; U_22[3, 6] = -1
        V_22[0, 6] = 1; V_22[2, 6] = 1
        W_22[0, 6] = 1  # C00
        
        # Now map 2x2 blocks to 4x4.
        # 4x4 is composed of four 2x2 blocks:
        # A = [[A00, A01], [A10, A11]]
        # A00 is rows 0-1, cols 0-1.
        # A01 is rows 0-1, cols 2-3.
        # A10 is rows 2-3, cols 0-1.
        # A11 is rows 2-3, cols 2-3.
        
        # We iterate over the 7 Strassen products (P0..P6).
        # Each product involves specific A and B blocks.
        # For each product, we instantiate the 2x2 Strassen logic.
        
        prod_defs = [
            # (A_blocks, B_blocks, C_blocks)
            # A_blocks: list of (bi, bj, coeff)
            # B_blocks: list of (bk, bl, coeff)
            # C_blocks: list of (ci, ck, coeff)
            
            # P0 (M1-like): (A00+A11)(B00+B11)
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            # P1 (M2-like): (A10+A11)B00
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]),
            # P2 (M3-like): A00(B01-B11)
            ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
            # P3 (M4-like): A11(B10-B00)
            ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
            # P4 (M5-like): (A00+A01)B11
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
            # P5 (M6-like): (A10-A11)(B01+B11)
            ([(1,0,1), (1,1,-1)], [(0,1,1), (1,1,1)], [(1,1,1)]),
            # P6 (M7-like): (A01-A11)(B00+B10)
            ([(0,1,1), (1,1,-1)], [(0,0,1), (1,0,1)], [(0,0,1)]),
        ]
        
        for r in range(7):
            A_blocks, B_blocks, C_blocks = prod_defs[r]
            
            for s in range(7):
                global_rank = r * 7 + s
                
                # Accumulate U, V, W
                # For each local index in 2x2 (0..3), we map to global 4x4 indices.
                # Local A index l_a (0..3) corresponds to:
                # l_a=0 -> A00[0,0] -> Global (0,0)
                # l_a=1 -> A00[0,1] -> Global (0,1)
                # l_a=2 -> A00[1,0] -> Global (1,0)
                # l_a=3 -> A00[1,1] -> Global (1,1)
                # Wait, the 2x2 Strassen operates on the *blocks*.
                # So "A00" in the 2x2 Strassen context refers to the entire 2x2 block A00 of the 4x4 matrix.
                # So if U_22[l_a, s] is non-zero, it means the term involves the block A_{block_a}.
                # We must add this coefficient to ALL elements in that block.
                
                # Mapping local 2x2 block index (0..3) to global 4x4 block:
                # 0 -> Block (0,0)
                # 1 -> Block (0,1)
                # 2 -> Block (1,0)
                # 3 -> Block (1,1)
                
                # For U (A factors):
                for l_a in range(4):
                    coeff = U_22[l_a, s]
                    if coeff == 0: continue
                    # Find which block this local index belongs to
                    # Actually, U_22 is defined for the 2x2 matrix of blocks.
                    # So l_a=0 is Block A00. l_a=1 is Block A01. l_a=2 is Block A10. l_a=3 is Block A11.
                    # We need to check if this block is in A_blocks.
                    # But A_blocks is a list of (bi, bj, coeff).
                    # The Strassen algorithm *is* the definition of the blocks.
                    # So we just use the A_blocks list to determine which global blocks are involved.
                    # Wait, no. The 2x2 Strassen factors U_22 tell us *which* local blocks are used.
                    # E.g. for M1, U_22 has 1s at 0 and 3. This means it uses Block A00 and Block A11.
                    # So we iterate over l_a, and if U_22[l_a, s] != 0, we know it uses the block corresponding to l_a.
                    # We then look up the coefficient for that block in A_blocks.
                    
                    # Map l_a to block indices (bi, bj)
                    bi = l_a // 2
                    bj = l_a % 2
                    
                    # Find coefficient in A_blocks
                    a_coeff = 0
                    for (bi_b, bj_b, c) in A_blocks:
                        if bi_b == bi and bj_b == bj:
                            a_coeff = c
                            break
                    if a_coeff == 0: continue
                    
                    total_coeff = coeff * a_coeff
                    if total_coeff == 0: continue
                    
                    # Add to all elements in block (bi, bj) of A
                    for i_loc in range(2):
                        for j_loc in range(2):
                            i_glob = 2*bi + i_loc
                            j_glob = 2*bj + j_loc
                            U_44[i_glob * 4 + j_glob, global_rank] += total_coeff
                
                # For V (B factors):
                for l_b in range(4):
                    coeff = V_22[l_b, s]
                    if coeff == 0: continue
                    bk = l_b // 2
                    bl = l_b % 2
                    
                    b_coeff = 0
                    for (bk_b, bl_b, c) in B_blocks:
                        if bk_b == bk and bl_b == bl:
                            b_coeff = c
                            break
                    if b_coeff == 0: continue
                    
                    total_coeff = coeff * b_coeff
                    if total_coeff == 0: continue
                    
                    for k_loc in range(2):
                        for l_loc in range(2):
                            k_glob = 2*bk + k_loc
                            l_glob = 2*bl + l_loc
                            V_44[k_glob * 4 + l_glob, global_rank] += total_coeff
                            
                # For W (C factors):
                # W_22 is for C blocks.
                # l_c=0 -> C00, l_c=1 -> C01, l_c=2 -> C10, l_c=3 -> C11
                # Wait, W_22 indices were:
                # C00 -> 0, C10 -> 1, C01 -> 2, C11 -> 3 (based on W[k*n+i] with n=2)
                # Let's re-verify W_22 indices used above.
                # M1: W[0], W[3] -> C00, C11.
                # M2: W[1] -> C10.
                # M3: W[2], W[3] -> C01, C11.
                # M4: W[0], W[1] -> C00, C10.
                # M5: W[0], W[2] -> C00, C01.
                # M6: W[3] -> C11.
                # M7: W[0] -> C00.
                
                # So l_c mapping:
                # 0 -> C00 (Block 0,0)
                # 1 -> C10 (Block 1,0)
                # 2 -> C01 (Block 0,1)
                # 3 -> C11 (Block 1,1)
                
                for l_c in range(4):
                    coeff = W_22[l_c, s]
                    if coeff == 0: continue
                    
                    # Map l_c to block (ci, ck)
                    # 0: (0,0), 1: (1,0), 2: (0,1), 3: (1,1)
                    if l_c == 0: ci, ck = 0, 0
                    elif l_c == 1: ci, ck = 1, 0
                    elif l_c == 2: ci, ck = 0, 1
                    else: ci, ck = 1, 1
                    
                    c_coeff = 0
                    for (ci_b, ck_b, c) in C_blocks:
                        if ci_b == ci and ck_b == ck:
                            c_coeff = c
                            break
                    if c_coeff == 0: continue
                    
                    total_coeff = coeff * c_coeff
                    if total_coeff == 0: continue
                    
                    for i_loc in range(2):
                        for k_loc in range(2):
                            i_glob = 2*ci + i_loc
                            k_glob = 2*ck + k_loc
                            # W index for 4x4: k_glob * 4 + i_glob
                            W_44[k_glob * 4 + i_glob, global_rank] += total_coeff
                            
        return U_44, V_44, W_44

    # Initialize full 5x5 factors
    total_rank = 110
    U = np.zeros((25, total_rank))
    V = np.zeros((25, total_rank))
    W = np.zeros((25, total_rank))
    
    # 1. A44 * B44 (Rank 49)
    U_44, V_44, W_44 = get_strassen_4x4_factors()
    # Place in top-left 4x4 blocks
    # U rows 0..15, V cols 0..15, W rows 0..15 (but W is permuted)
    # U_44 is (16, 49). U is (25, 110).
    # Rows 0..15 of U correspond to A[0:4, 0:4].
    # Cols 0..48 of U correspond to the 49 terms.
    U[0:16, 0:49] = U_44
    V[0:16, 0:49] = V_44
    W[0:16, 0:49] = W_44
    
    # 2. A41 * B14 (Rank 16)
    # A41: rows 0..3, col 4. B14: row 4, cols 0..3. C41: rows 0..3, col 4.
    # n_sub=4, m_sub=1, p_sub=4.
    # off_i=0, off_j=4. off_k=4, off_l=0. off_ci=0, off_ck=4.
    rank_used = add_schoolblock(4, 1, 4, 0, 4, 4, 0, 0, 4, 49, U, V, W)
    
    # 3. A44 * B41 (Rank 16)
    # A44: rows 0..3, cols 0..3. B41: rows 0..3, col 4. C41: rows 0..3, col 4.
    # n_sub=4, m_sub=4, p_sub=1.
    # off_i=0, off_j=0. off_k=0, off_l=4. off_ci=0, off_ck=4.
    rank_used = add_schoolblock(4, 4, 1, 0, 0, 0, 4, 0, 4, 49+16, U, V, W)
    
    # 4. A41 * B11 (Rank 4)
    # A41: rows 0..3, col 4. B11: row 4, col 4. C41: rows 0..3, col 4.
    # n_sub=4, m_sub=1, p_sub=1.
    # off_i=0, off_j=4. off_k=4, off_l=4. off_ci=0, off_ck=4.
    rank_used = add_schoolblock(4, 1, 1, 0, 4, 4, 4, 0, 4, 49+16+16, U, V, W)
    
    # 5. A14 * B44 (Rank 16)
    # A14: row 4, cols 0..3. B44: rows 0..3, cols 0..3. C14: row 4, cols 0..3.
    # n_sub=1, m_sub=4, p_sub=4.
    # off_i=4, off_j=0. off_k=0, off_l=0. off_ci=4, off_ck=0.
    rank_used = add_schoolblock(1, 4, 4, 4, 0, 0, 0, 4, 0, 49+16+16+4, U, V, W)
    
    # 6. A11 * B14 (Rank 4)
    # A11: row 4, col 4. B14: row 4, cols 0..3. C14: row 4, cols 0..3.
    # n_sub=1, m_sub=1, p_sub=4.
    # off_i=4, off_j=4. off_k=4, off_l=0. off_ci=4, off_ck=0.
    rank_used = add_schoolblock(1, 1, 4, 4, 4, 4, 0, 4, 0, 49+16+16+4+16, U, V, W)
    
    # 7. A14 * B41 (Rank 4)
    # A14: row 4, cols 0..3. B41: rows 0..3, col 4. C14: row 4, col 4.
    # n_sub=1, m_sub=4, p_sub=1.
    # off_i=4, off_j=0. off_k=0, off_l=4. off_ci=4, off_ck=4.
    rank_used = add_schoolblock(1, 4, 1, 4, 0, 0, 4, 4, 4, 49+16+16+4+16+4, U, V, W)
    
    # 8. A11 * B11 (Rank 1)
    # A11: row 4, col 4. B11: row 4, col 4. C11: row 4, col 4.
    # n_sub=1, m_sub=1, p_sub=1.
    # off_i=4, off_j=4. off_k=4, off_l=4. off_ci=4, off_ck=4.
    rank_used = add_schoolblock(1, 1, 1, 4, 4, 4, 4, 4, 4, 49+16+16+4+16+4+4, U, V, W)
    
    # Verify
    if time.time() < deadline - 5:
        np.random.seed(42)
        for _ in range(10):
            A = np.random.randint(-5, 5, (n, m))
            B = np.random.randint(-5, 5, (m, p))
            C_expected = A @ B
            
            C_calc = np.zeros((n, p))
            for r in range(total_rank):
                alpha = np.zeros(n)
                for i in range(n):
                    for j in range(m):
                        alpha[i] += U[i*m+j, r] * A[i,j]
                beta = np.zeros(p)
                for k in range(p):
                    for j in range(m):
                        beta[k] += V[j*p+k, r] * B[j,k]
                
                for i in range(n):
                    for k in range(p):
                        C_calc[i,k] += alpha[i] * beta[k] * W[k*n+i, r]
                        
            if not np.allclose(C_calc, C_expected, atol=1e-5):
                # If verification fails, fallback to schoolbook
                # This should not happen if the construction is correct.
                # But for safety, we return schoolbook.
                rank = n * m * p
                U_f = np.zeros((n * m, rank))
                V_f = np.zeros((m * p, rank))
                W_f = np.zeros((p * n, rank))
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_f[i * m + j, col] = 1
                            V_f[j * p + k, col] = 1
                            W_f[k * n + i, col] = 1
                            col += 1
                return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

