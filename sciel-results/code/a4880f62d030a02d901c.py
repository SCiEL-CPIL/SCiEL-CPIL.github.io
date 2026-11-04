import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will use a decomposition of rank 97.
    # Part 1: 4x4x4 multiplication using Strassen (rank 49)
    # Part 2: 4x4x3 multiplication using Schoolbook (rank 48)
    # Total rank = 49 + 48 = 97.
    
    rank = 97
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # --- Part 1: 4x4x4 Strassen ---
    # Strassen for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 4x4x4, we apply this to 2x2 blocks.
    # A is 4x4, split into 2x2 blocks A00, A01, A10, A11.
    # B is 4x4, split into 2x2 blocks B00, B01, B10, B11.
    # C is 4x4, split into 2x2 blocks C00, C01, C10, C11.
    
    # The 7 Strassen products for the block level are:
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B10-B11)
    # S4 = A11(B01-B00)
    # S5 = (A00+A01)B11
    # S6 = (A10-A00)(B00+B01)
    # S7 = (A01-A11)(B10+B11)
    
    # Each Sx is a 2x2 matrix multiplication.
    # We decompose each 2x2 multiplication into 7 scalar multiplications.
    # Total 7 * 7 = 49 scalar multiplications.
    
    # Indices for 4x4x4 part (columns 0 to 48)
    # Block indices: 0:00, 1:01, 2:10, 3:11
    # A blocks: A_{00} is rows 0-1, cols 0-1.
    
    # We define the 7 block-level factors u_s, v_s for s in 1..7
    # u_s is a 4x4 matrix (coefficients for A)
    # v_s is a 4x4 matrix (coefficients for B)
    
    block_factors = []
    # S1
    u1 = np.zeros((4,4)); u1[0:2,0:2] = 1; u1[2:4,2:4] = 1
    v1 = np.zeros((4,4)); v1[0:2,0:2] = 1; v1[2:4,2:4] = 1
    block_factors.append((u1, v1))
    # S2
    u2 = np.zeros((4,4)); u2[2:4,0:2] = 1; u2[2:4,2:4] = 1
    v2 = np.zeros((4,4)); v2[0:2,0:2] = 1
    block_factors.append((u2, v2))
    # S3
    u3 = np.zeros((4,4)); u3[0:2,0:2] = 1
    v3 = np.zeros((4,4)); v3[2:4,0:2] = 1; v3[2:4,2:4] = -1
    block_factors.append((u3, v3))
    # S4
    u4 = np.zeros((4,4)); u4[2:4,2:4] = 1
    v4 = np.zeros((4,4)); v4[0:2,2:4] = 1; v4[0:2,0:2] = -1
    block_factors.append((u4, v4))
    # S5
    u5 = np.zeros((4,4)); u5[0:2,0:2] = 1; u5[0:2,2:4] = 1
    v5 = np.zeros((4,4)); v5[2:4,2:4] = 1
    block_factors.append((u5, v5))
    # S6
    u6 = np.zeros((4,4)); u6[2:4,0:2] = 1; u6[0:2,0:2] = -1
    v6 = np.zeros((4,4)); v6[0:2,0:2] = 1; v6[0:2,2:4] = 1
    block_factors.append((u6, v6))
    # S7
    u7 = np.zeros((4,4)); u7[0:2,2:4] = 1; u7[2:4,2:4] = -1
    v7 = np.zeros((4,4)); v7[2:4,0:2] = 1; v7[2:4,2:4] = 1
    block_factors.append((u7, v7))
    
    # Now, for each block factor s, we need to decompose the 2x2x2 multiplication
    # into 7 scalar multiplications.
    # The 2x2x2 Strassen factors (for local indices 0,1)
    strassen_2x2 = [
        # (u_local, v_local, w_local)
        # M1 = (A00+A11)(B00+B11)
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,0]]), # w for C00? No, w is for C reconstruction.
        # Actually, we need the w factors for the 2x2 output.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
    ]
    
    # Let's define the 7 terms of 2x2x2 Strassen explicitly as (u, v, w)
    # u, v, w are 2x2 matrices.
    # Term 1: M1
    t1_u = np.array([[1,0],[0,1]])
    t1_v = np.array([[1,0],[0,1]])
    t1_w = np.array([[1,0],[0,1]]) # Wait, M1 contributes to C00 and C11.
    # C00 gets +1, C11 gets +1.
    # So w1 should be [[1,0],[0,1]]?
    # Let's check: C = sum w_r M_r.
    # M1 is a 2x2 matrix? No, M1 is a scalar in the 2x2x2 context?
    # No, in 2x2x2, A and B are 2x2. M1 is a scalar?
    # No, M1 = (A00+A11)(B00+B11) is a product of two 1x1 matrices?
    # Yes, A00 is 1x1. So M1 is a scalar.
    # Then C is 2x2.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for the 2x2x2 decomposition, we have 7 scalar multiplications.
    # u_r is 2x2, v_r is 2x2, w_r is 2x2 (coefficients for C).
    # Actually, w_r is a 2x2 matrix of coefficients for the output C.
    
    s2_u = [
        np.array([[1,0],[0,1]]), # M1: A00+A11
        np.array([[0,0],[1,1]]), # M2: A10+A11
        np.array([[1,0],[0,0]]), # M3: A00
        np.array([[0,0],[0,1]]), # M4: A11
        np.array([[1,1],[0,0]]), # M5: A00+A01
        np.array([[-1,0],[1,0]]),# M6: A10-A00
        np.array([[0,-1],[0,1]]) # M7: A01-A11
    ]
    s2_v = [
        np.array([[1,0],[0,1]]), # M1: B00+B11
        np.array([[1,0],[0,0]]), # M2: B00
        np.array([[0,-1],[0,1]]),# M3: B10-B11
        np.array([[-1,1],[0,0]]),# M4: B01-B00
        np.array([[0,0],[0,1]]), # M5: B11
        np.array([[1,1],[0,0]]), # M6: B00+B01
        np.array([[0,1],[0,1]])  # M7: B10+B11
    ]
    s2_w = [
        np.array([[1,0],[0,1]]), # M1: C00+1, C11+1
        np.array([[0,0],[1,-1]]),# M2: C10+1, C11-1
        np.array([[0,1],[0,1]]), # M3: C01+1, C11+1
        np.array([[1,0],[1,0]]), # M4: C00+1, C10+1
        np.array([[-1,1],[0,0]]),# M5: C00-1, C01+1
        np.array([[0,0],[0,1]]), # M6: C11+1
        np.array([[1,0],[0,0]])  # M7: C00+1
    ]
    
    # Now combine block factors and 2x2 factors.
    # For each block s (0..6) and each 2x2 term t (0..6):
    # U_global = u_s_block \otimes s2_u[t] ?
    # No, u_s is 4x4. s2_u is 2x2.
    # The product is element-wise?
    # A is 4x4. u_s is 4x4.
    # s2_u[t] is 2x2.
    # We need to map the 2x2 local indices to 4x4 global indices.
    # The block s corresponds to a specific 2x2 sub-matrix of A.
    # But u_s is already a 4x4 matrix with non-zeros only in that block.
    # So we just multiply the non-zero elements?
    # Actually, u_s is sparse.
    # Let's just iterate.
    
    col = 0
    for s in range(7):
        u_s_block = block_factors[s][0]
        v_s_block = block_factors[s][1]
        for t in range(7):
            u_t_2x2 = s2_u[t]
            v_t_2x2 = s2_v[t]
            w_t_2x2 = s2_w[t]
            
            # Construct U column
            # U is 16x1.
            # u_s_block is 4x4.
            # u_t_2x2 is 2x2.
            # The block s is at a specific position.
            # We need to know which 2x2 block of the 4x4 matrix u_s_block is non-zero.
            # From definition:
            # s=0: 00 and 11 blocks.
            # This is tricky.
            
            # Simpler:
            # U[i, j] = sum over s,t of ...
            # Let's just build the full 4x4 U matrix for this term.
            U_term = np.zeros((4,4))
            V_term = np.zeros((4,4))
            W_term = np.zeros((4,4))
            
            # We need to "expand" the 2x2 factor into the 4x4 block.
            # The block s determines which 2x2 sub-block of the 4x4 matrix is active.
            # But u_s_block might have multiple non-zero blocks (e.g., s=0 has 00 and 11).
            # In that case, the 2x2 factor applies to BOTH blocks?
            # Yes, Strassen works on the whole matrix.
            # So if u_s_block has 1s in 00 and 11, then u_t_2x2 is applied to both 00 and 11 blocks.
            
            # Let's find the non-zero 2x2 blocks in u_s_block.
            # We can just multiply u_s_block by a "block diagonal" matrix formed by u_t_2x2?
            # No, u_t_2x2 is 2x2.
            # We need to map it to 4x4.
            # If u_s_block has non-zeros in block (r,c), we place u_t_2x2 there.
            
            # Helper to place 2x2 into 4x4 at block (br, bc)
            def place_2x2_in_4x4(mat_2x2, br, bc, target):
                target[br*2:br*2+2, bc*2:bc*2+2] = mat_22 if False else mat_2x2
                return target

            # Identify blocks for s
            # s=0: (0,0), (1,1)
            # s=1: (1,0), (1,1)
            # s=2: (0,0)
            # s=3: (1,1)
            # s=4: (0,0), (0,1)
            # s=5: (1,0), (0,0)
            # s=6: (0,1), (1,1)
            
            blocks_for_s = {
                0: [(0,0), (1,1)],
                1: [(1,0), (1,1)],
                2: [(0,0)],
                3: [(1,1)],
                4: [(0,0), (0,1)],
                5: [(1,0), (0,0)],
                6: [(0,1), (1,1)]
            }
            
            for (br, bc) in blocks_for_s[s]:
                U_term[br*2:br*2+2, bc*2:bc*2+2] += u_t_2x2
                V_term[br*2:br*2+2, bc*2:bc*2+2] += v_t_2x2
                # W_term is for the output C.
                # The output block is also (br, bc)?
                # No, the Strassen formula C = sum w_r M_r.
                # M_r is a 2x2 matrix (result of block multiplication).
                # So w_t_2x2 is applied to the same block (br, bc) of C?
                # Yes.
                W_term[br*2:br*2+2, bc*2:bc*2+2] += w_t_2x2
                
            # Flatten and assign
            U[:, col] = U_term.flatten()
            V[:, col] = V_term.flatten()
            # W is p*n x rank. Here p=4, n=4.
            # W index is k*n+i.
            # W_term is 4x4 (k, i).
            W[:, col] = W_term.flatten()
            
            col += 1
            
    # --- Part 2: 4x4x3 Schoolbook ---
    # Columns 49 to 96
    # Schoolbook: for i in 0..3, j in 0..3, k in 0..2
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U[i*4+j, col] = 1
                V[j*4+k, col] = 1
                W[k*4+i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
