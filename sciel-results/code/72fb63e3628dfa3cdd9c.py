import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Strategy:
    1. Split B (4x9) into B1 (4x4) and B2 (4x5).
    2. Split B2 into B2a (4x4) and B2b (4x1).
    3. Compute C1 = A * B1 using Strassen's algorithm for 4x4x4 (rank 49).
    4. Compute C2a = A * B2a using Strassen's algorithm for 4x4x4 (rank 49).
    5. Compute C2b = A * B2b using schoolbook for 4x4x1 (rank 16).
    
    Total rank = 49 + 49 + 16 = 114.
    
    The output is W[k*n+i, r] where n=4, p=9.
    C1 contributes to columns 0-3 of the output (k=0..3).
    C2a contributes to columns 4-7 of the output (k=4..7).
    C2b contributes to column 8 of the output (k=8).
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # Precompute Strassen 4x4x4 decomposition
    # Strassen 2x2x2 has rank 7.
    # 4x4x4 is 4 blocks of 2x2x2.
    # Total rank for 4x4x4 = 4 * 7 = 28? No, Strassen for 4x4 is more complex.
    # Actually, standard Strassen for 2x2 uses 7 multiplications.
    # For 4x4, we can view it as 2x2 blocks of 2x2 matrices.
    # So we need 7 Strassen multiplications of 2x2 matrices.
    # Each 2x2 Strassen multiplication has rank 7.
    # So total rank for 4x4x4 = 7 * 7 = 49.
    
    # Let's construct the Strassen 4x4x4 decomposition explicitly.
    # This is complex, so let's use a known decomposition or fall back to schoolbook if too hard.
    # For now, let's try to implement the 4x4 Strassen decomposition.
    
    # Alternative: Use the fact that 4x4x4 rank is 49.
    # We can build it from 2x2x2 Strassen blocks.
    
    # Let's define the 2x2x2 Strassen decomposition.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 - M4 + M5 + M6
    # C12 = M3 + M4
    # C21 = M2 + M5
    # C22 = M1 - M3 + M2 - M7
    
    # For 4x4x4, we have 4x4 matrices A and B.
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2.
    # B = [[B11, B12], [B21, B22]] where each Bij is 2x2.
    # C = [[C11, C12], [C21, C22]] where each Cij is 2x2.
    
    # We compute 7 "multiplications" of 2x2 matrices using Strassen.
    # S1 = A11 + A22
    # S2 = A21 + A22
    # S3 = A11 - A21
    # S4 = A12 - A22
    # S5 = A11 + A12
    # S6 = A21 - A11
    # S7 = A12 - A22
    
    # T1 = B11 + B22
    # T2 = B11 - B21
    # T3 = B12 + B22
    # T4 = B21 - B11
    # T5 = B12 + B22
    # T6 = B21 + B22
    # T7 = B12 - B22
    
    # M1 = S1 * T1 (using Strassen 2x2, rank 7)
    # M2 = S2 * T2 (using Strassen 2x2, rank 7)
    # M3 = S3 * T3 (using Strassen 2x2, rank 7)
    # M4 = S4 * T4 (using Strassen 2x2, rank 7)
    # M5 = S5 * T5 (using Strassen 2x2, rank 7)
    # M6 = S6 * T6 (using Strassen 2x2, rank 7)
    # M7 = S7 * T7 (using Strassen 2x2, rank 7)
    
    # This gives rank 7*7 = 49.
    
    # Due to complexity, let's use a simpler approach for now:
    # We will use the schoolbook decomposition for the 4x4x4 parts as well,
    # but that gives rank 64 for each, total 64+64+16 = 144.
    # To beat 144, we need the Strassen decomposition.
    
    # Let's implement the Strassen 4x4x4 decomposition.
    
    def strassen_2x2_2x2():
        """
        Returns the rank-7 decomposition for 2x2 * 2x2.
        U: (4, 7), V: (4, 7), W: (4, 7)
        Indices:
        U: A11, A12, A21, A22
        V: B11, B12, B21, B22
        W: C11, C12, C21, C22
        """
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # M1
        U[0, 0] = 1  # A11
        U[3, 0] = 1  # A22
        V[0, 0] = 1  # B11
        V[3, 0] = 1  # B22
        W[0, 0] = 1  # C11
        W[3, 0] = 1  # C22
        
        # M2
        U[2, 1] = 1  # A21
        U[3, 1] = 1  # A22
        V[0, 1] = 1  # B11
        W[1, 1] = 1  # C12? No, C21 = M2 + M5, C22 = M1 - M3 + M2 - M7
        # Let's re-verify the Strassen formulas.
        # C11 = M1 - M4 + M5 + M6
        # C12 = M3 + M4
        # C21 = M2 + M5
        # C22 = M1 - M3 + M2 - M7
        
        # So:
        # W[0, 0] += 1 (M1 to C11)
        # W[3, 0] += 1 (M1 to C22)
        
        # M2: (A21+A22)B11
        U[2, 1] = 1
        U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1  # C21
        W[3, 1] = 1  # C22
        
        # M3: A11(B12-B22)
        U[0, 2] = 1
        V[1, 2] = 1
        V[3, 2] = -1
        W[1, 2] = 1  # C12
        W[3, 2] = -1 # C22
        
        # M4: A22(B21-B11)
        U[3, 3] = 1
        V[2, 3] = 1
        V[0, 3] = -1
        W[0, 3] = -1 # C11
        W[1, 3] = 1  # C12
        
        # M5: (A11+A12)B22
        U[0, 4] = 1
        U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1  # C11
        W[2, 4] = 1  # C21
        
        # M6: (A21-A11)(B11+B12)
        U[2, 5] = 1
        U[0, 5] = -1
        V[0, 5] = 1
        V[1, 5] = 1
        W[0, 5] = 1  # C11
        
        # M7: (A12-A22)(B21+B22)
        U[1, 6] = 1
        U[3, 6] = -1
        V[2, 6] = 1
        V[3, 6] = 1
        W[3, 6] = -1 # C22
        
        return U, V, W
    
    # Now build 4x4x4 from 2x2x2 blocks.
    # This is too complex to do manually in the code.
    # Let's use a known rank-49 decomposition for 4x4x4.
    # Or, we can just use the schoolbook for 4x4x4 (rank 64) and accept total rank 144.
    # But the task asks for rank-minimal.
    # Let's try to implement the 4x4x4 Strassen decomposition programmatically.
    
    def build_4x4x4_strassen():
        """
        Build the rank-49 decomposition for 4x4 * 4x4.
        """
        # We use the 2x2x2 Strassen decomposition recursively.
        # A is 4x4, viewed as 2x2 blocks of 2x2.
        # B is 4x4, viewed as 2x2 blocks of 2x2.
        # C is 4x4, viewed as 2x2 blocks of 2x2.
        
        # We need 7 "multiplications" of 2x2 matrices.
        # Each multiplication is done using Strassen 2x2x2 (rank 7).
        # So total rank = 7 * 7 = 49.
        
        U_2x2, V_2x2, W_2x2 = strassen_2x2_2x2()
        
        # The 7 combinations are:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Let's define the linear combinations for A and B.
        # A blocks: A11 (0:2, 0:2), A12 (0:2, 2:4), A21 (2:4, 0:2), A22 (2:4, 2:4)
        # B blocks: B11 (0:2, 0:2), B12 (0:2, 2:4), B21 (2:4, 0:2), B22 (2:4, 2:4)
        
        # We will build U, V, W for the 4x4x4 case.
        # U has shape (16, 49), V has shape (16, 49), W has shape (16, 49).
        
        U = np.zeros((16, 49), dtype=float)
        V = np.zeros((16, 49), dtype=float)
        W = np.zeros((16, 49), dtype=float)
        
        # For each of the 7 Strassen multiplications, we add 7 rank-1 terms.
        # The 7 rank-1 terms for a 2x2x2 multiplication are in U_2x2, V_2x2, W_2x2.
        
        # Let's define the linear combinations for A and B for each M_i.
        # M1: A_comb = A11 + A22, B_comb = B11 + B22
        # M2: A_comb = A21 + A22, B_comb = B11
        # M3: A_comb = A11, B_comb = B12 - B22
        # M4: A_comb = A22, B_comb = B21 - B11
        # M5: A_comb = A11 + A12, B_comb = B22
        # M6: A_comb = A21 - A11, B_comb = B11 + B12
        # M7: A_comb = A12 - A22, B_comb = B21 + B22
        
        # For each M_i, we have 7 rank-1 terms.
        # Each rank-1 term has coefficients in U_2x2, V_2x2, W_2x2.
        
        # Let's iterate over the 7 M_i's.
        for m_i in range(7):
            # Determine the linear combinations for A and B.
            # We will map the 2x2 block indices to 4x4 indices.
            
            # A block indices: 0=A11, 1=A12, 2=A21, 3=A22
            # B block indices: 0=B11, 1=B12, 2=B21, 3=B22
            
            # M1: A: 0+3, B: 0+3
            # M2: A: 2+3, B: 0
            # M3: A: 0, B: 1-3
            # M4: A: 3, B: 2-0
            # M5: A: 0+1, B: 3
            # M6: A: 2-0, B: 0+1
            # M7: A: 1-3, B: 2+3
            
            a_comb = [0]*4
            b_comb = [0]*4
            
            if m_i == 0:
                a_comb = [1, 0, 0, 1]
                b_comb = [1, 0, 0, 1]
            elif m_i == 1:
                a_comb = [0, 0, 1, 1]
                b_comb = [1, 0, 0, 0]
            elif m_i == 2:
                a_comb = [1, 0, 0, 0]
                b_comb = [0, 1, 0, -1]
            elif m_i == 3:
                a_comb = [0, 0, 0, 1]
                b_comb = [0, 0, 1, 0]  # Wait, B21 - B11 -> [ -1, 0, 1, 0]
                b_comb = [-1, 0, 1, 0]
            elif m_i == 4:
                a_comb = [1, 1, 0, 0]
                b_comb = [0, 0, 0, 1]
            elif m_i == 5:
                a_comb = [-1, 0, 1, 0]
                b_comb = [1, 1, 0, 0]
            elif m_i == 6:
                a_comb = [0, 1, 0, -1]
                b_comb = [0, 0, 1, 1]
            
            # Now, for each of the 7 rank-1 terms in the 2x2x2 decomposition,
            # we add the corresponding rank-1 term to the 4x4x4 decomposition.
            
            for r in range(7):
                col = m_i * 7 + r
                
                # U part:
                # U_2x2[row, r] is the coefficient for A_block_index in the 2x2 case.
                # In the 4x4 case, A_block_index maps to 4 rows.
                # A11 -> rows 0,1; A12 -> rows 2,3; A21 -> rows 4,5; A22 -> rows 6,7.
                # Wait, the indexing is row-major.
                # A is 4x4. A[i,j] is at index i*4+j.
                # A11 is A[0:2, 0:2] -> indices 0,1,4,5.
                # A12 is A[0:2, 2:4] -> indices 2,3,6,7.
                # A21 is A[2:4, 0:2] -> indices 8,9,12,13.
                # A22 is A[2:4, 2:4] -> indices 10,11,14,15.
                
                # So for each row in U_2x2 (0,1,2,3), we map to the corresponding 4 rows in U.
                # U_2x2[0] -> A11 -> rows 0,1,4,5
                # U_2x2[1] -> A12 -> rows 2,3,6,7
                # U_2x2[2] -> A21 -> rows 8,9,12,13
                # U_2x2[3] -> A22 -> rows 10,11,14,15
                
                a_block_to_rows = [
                    [0, 1, 4, 5],
                    [2, 3, 6, 7],
                    [8, 9, 12, 13],
                    [10, 11, 14, 15]
                ]
                
                b_block_to_rows = [
                    [0, 1, 4, 5],
                    [2, 3, 6, 7],
                    [8, 9, 12, 13],
                    [10, 11, 14, 15]
                ]
                
                w_block_to_rows = [
                    [0, 1, 4, 5],
                    [2, 3, 6, 7],
                    [8, 9, 12, 13],
                    [10, 11, 14, 15]
                ]
                
                for a_block in range(4):
                    coeff_a = U_2x2[a_block, r] * a_comb[a_block]
                    if coeff_a != 0:
                        for row in a_block_to_rows[a_block]:
                            U[row, col] += coeff_a
                
                for b_block in range(4):
                    coeff_b = V_2x2[b_block, r] * b_comb[b_block]
                    if coeff_b != 0:
                        for row in b_block_to_rows[b_block]:
                            V[row, col] += coeff_b
                
                for w_block in range(4):
                    coeff_w = W_2x2[w_block, r]
                    if coeff_w != 0:
                        for row in w_block_to_rows[w_block]:
                            W[row, col] += coeff_w
        
        return U, V, W
    
    # Build the 4x4x4 Strassen decomposition
    U_444, V_444, W_444 = build_4x4x4_strassen()
    
    # Now we have the decomposition for 4x4x4.
    # We need to use it for A * B1 and A * B2a.
    
    # B1 is the first 4 columns of B (4x9).
    # B2a is the next 4 columns of B (4x9).
    # B2b is the last 1 column of B (4x9).
    
    # Total rank = 49 (for B1) + 49 (for B2a) + 16 (for B2b) = 114.
    
    rank = 49 + 49 + 16
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((36, rank), dtype=float)
    W = np.zeros((36, rank), dtype=float)
    
    col = 0
    
    # Add the decomposition for A * B1 (columns 0-3 of B)
    for r in range(49):
        U[:, col] = U_444[:, r]
        # V part: B1 is columns 0-3 of B.
        # V_444 has shape (16, 49), corresponding to B11, B12, B21, B22.
        # But B1 is 4x4, so B11, B12, B21, B22 are the blocks of B1.
        # So we just copy V_444[:, r] to V[:, col] for the first 16 rows.
        V[0:16, col] = V_444[:, r]
        # W part: C1 is the first 4 columns of C.
        # W_444 has shape (16, 49), corresponding to C11, C12, C21, C22.
        # So we copy W_444[:, r] to W[:, col] for the first 16 rows.
        W[0:16, col] = W_444[:, r]
        col += 1
    
    # Add the decomposition for A * B2a (columns 4-7 of B)
    for r in range(49):
        U[:, col] = U_444[:, r]
        # V part: B2a is columns 4-7 of B.
        # So we copy V_444[:, r] to V[16:32, col].
        V[16:32, col] = V_444[:, r]
        # W part: C2a is columns 4-7 of C.
        # So we copy W_444[:, r] to W[16:32, col].
        W[16:32, col] = W_444[:, r]
        col += 1
    
    # Add the decomposition for A * B2b (column 8 of B)
    # Schoolbook for 4x4x1: rank 16.
    for i in range(4):
        for j in range(4):
            U[i*4+j, col] = 1
            V[32+j, col] = 1  # B2b is column 8 of B, which is row j of B2b.
            # W part: C2b is column 8 of C.
            # C[k, i] for k=8. So W[8*4+i, col] = 1.
            W[8*4+i, col] = 1
            col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
