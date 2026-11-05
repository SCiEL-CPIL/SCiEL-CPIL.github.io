import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Part 1: 4x4x4 using Strassen (rank 49)
    # Part 2: 4x4x3 using Schoolbook (rank 48)
    rank1 = 49
    rank2 = 48
    total_rank = rank1 + rank2
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # --- Part 1: 4x4x4 Strassen ---
    # Indices for U: i*m + j
    # Indices for V: j*p + k (but here p is effectively 4 for this part)
    # Indices for W: k*n + i
    
    # We need the rank-7 Strassen decomposition for 2x2x2.
    # Let's define the U, V, W for a single 2x2x2 Strassen block.
    # U (2*2, 7), V (2*2, 7), W (2*2, 7)
    
    # Strassen factors for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U_strassen (4, 7)
    # Rows: 00, 01, 10, 11
    U_s = np.zeros((4, 7))
    # V_strassen (4, 7)
    V_s = np.zeros((4, 7))
    # W_strassen (4, 7)
    W_s = np.zeros((4, 7))
    
    # M1
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    # M2
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = 1
    # M3
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[0, 2] = 1; W_s[1, 2] = 1
    # M4
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[2, 3] = 1; W_s[3, 3] = 1
    # M5
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = 1; W_s[1, 4] = 1
    # M6
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[2, 5] = 1; W_s[3, 5] = 1
    # M7
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1; W_s[1, 6] = 1
    
    # Now we combine 7 such blocks to form the 4x4x4 decomposition.
    # The 4x4 matrices are divided into 2x2 blocks.
    # Block indices: (0,0), (0,1), (1,0), (1,1)
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # The 7 Strassen multiplications are on the 2x2 blocks.
    # For each of the 7 multiplications, we use the rank-7 decomposition.
    # Total rank 49.
    
    # We need to map the global indices to the local 2x2 block indices.
    # Global i, j in 0..3.
    # Block row br = i // 2, Block col bc = j // 2.
    # Local row lr = i % 2, Local col lc = j % 2.
    # Local index in 2x2: lr * 2 + lc.
    
    # For the 4x4x4 part, we iterate over the 7 Strassen steps.
    # Each step involves a linear combination of the 4 blocks of A and B.
    # Actually, it's easier to think of it as:
    # The rank-49 decomposition is the tensor product of the rank-7 decomposition with itself?
    # No, it's just applying Strassen to the 2x2 blocks.
    
    # Let's construct the U, V, W for the 4x4x4 part.
    # U_4 (16, 49)
    # V_4 (16, 49)
    # W_4 (16, 49)
    
    U_4 = np.zeros((16, 49))
    V_4 = np.zeros((16, 49))
    W_4 = np.zeros((16, 49))
    
    # The 7 Strassen steps correspond to specific linear combinations of the 4x4x4 tensor.
    # It is equivalent to having 7 "super-multiplications" of 2x2 blocks.
    # For each super-multiplication s (0..6), we have a rank-7 decomposition of the 2x2x2 product.
    # So for each s, we have 7 scalar multiplications.
    # Total 49 scalar multiplications.
    
    # For a given super-multiplication s, the "effective" A_block and B_block are linear combinations of the 4 blocks.
    # But we can just embed the U_s, V_s, W_s into the larger matrices.
    
    # For super-multiplication s, and inner Strassen step t (0..6):
    # The column index in the 4x4x4 decomposition is s * 7 + t.
    
    # We need to determine which global elements contribute to the "A_block" and "B_block" for each (s, t).
    # This is getting complicated to do manually. Let's use the property that Strassen is a bilinear map.
    
    # Alternative: Use the known rank-49 decomposition coefficients.
    # Or, simpler: Since we just need *a* valid decomposition of rank 49, we can use the recursive structure.
    
    # Let's verify the rank-7 decomposition of 2x2x2 first.
    # U_s, V_s, W_s are defined above.
    
    # For 4x4x4:
    # The tensor is $T_{ijk} = \sum_{l,m,n} A_{il} B_{jm} \delta_{kn}$? No, $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # $T_{ijk} = \delta_{ik}$? No.
    
    # Let's just use the fact that we can decompose the 4x4x4 tensor into 7 copies of the 2x2x2 tensor, where the inputs are linear combinations.
    # This is exactly what Strassen does.
    
    # Let's define the linear combinations for the 7 Strassen steps.
    # A_blocks: A00, A01, A10, A11
    # B_blocks: B00, B01, B10, B11
    
    # Step 0: (A00+A11)(B00+B11)
    # Step 1: (A10+A11)B00
    # Step 2: A00(B01-B11)
    # Step 3: A11(B10-B00)
    # Step 4: (A00+A01)B11
    # Step 5: (A10-A00)(B00+B01)
    # Step 6: (A01-A11)(B10+B11)
    
    # For each step s, we have a "virtual" 2x2 matrix A_s and B_s.
    # A_s is a linear combination of the 4 blocks of A.
    # B_s is a linear combination of the 4 blocks of B.
    # Then we multiply A_s and B_s using the rank-7 Strassen decomposition.
    
    # So, for each step s, we define matrices U_s_block (4x4), V_s_block (4x4) such that
    # A_s = U_s_block * A_flattened? No.
    # A_s is a 2x2 matrix. Its elements are linear combinations of the 16 elements of A.
    # Actually, A_s is formed by taking linear combinations of the 4 2x2 blocks.
    # So A_s[0,0] = sum_{br,bc} c_{s, br, bc, 0, 0} A[br, bc, 0, 0] ...
    
    # Let's define the coefficients for A_s and B_s.
    # A_s is a 2x2 matrix.
    # B_s is a 2x2 matrix.
    
    # Coefficients for A_s (s, lr, lc, br, bc)
    # Coefficients for B_s (s, lr, lc, br, bc)
    
    # Step 0: A0 = A00 + A11, B0 = B00 + B11
    # Step 1: A1 = A10 + A11, B1 = B00
    # Step 2: A2 = A00, B2 = B01 - B11
    # Step 3: A3 = A11, B3 = B10 - B00
    # Step 4: A4 = A00 + A01, B4 = B11
    # Step 5: A5 = A10 - A00, B5 = B00 + B01
    # Step 6: A6 = A01 - A11, B6 = B10 + B11
    
    # For each step s, we have a 2x2 matrix A_s and B_s.
    # We then apply the rank-7 Strassen decomposition to A_s * B_s.
    # The rank-7 decomposition uses U_s, V_s, W_s (4x7 each).
    # The input to U_s is the 4 elements of A_s.
    # The input to V_s is the 4 elements of B_s.
    # The output of W_s is the 4 elements of C_s.
    
    # So, for a fixed s, the contribution to the global U, V, W is:
    # U_global[:, s*7 + t] = sum_{lr,lc} U_s[lr*2+lc, t] * coeff_A[s, lr, lc, br, bc]
    # Wait, U_s acts on the 4 elements of A_s.
    # A_s[lr, lc] = sum_{br, bc} coeff_A[s, lr, lc, br, bc] * A[br*2+lr, bc*2+lc]
    
    # So U_global[i, s*7+t] = sum_{lr,lc} U_s[lr*2+lc, t] * coeff_A[s, lr, lc, i//4, i%4]
    # Where i is the global index in A (0..15).
    # i = br*4 + bc*2 + lc? No.
    # Global index in A: i_global = i_row * 4 + i_col.
    # i_row = br*2 + lr. i_col = bc*2 + lc.
    # So i_global = (br*2 + lr)*4 + (bc*2 + lc).
    
    # Let's precompute the coeff_A and coeff_B.
    
    coeff_A = np.zeros((7, 2, 2, 2, 2)) # s, lr, lc, br, bc
    coeff_B = np.zeros((7, 2, 2, 2, 2))
    
    # Step 0: A0 = A00 + A11
    coeff_A[0, :, :, 0, 0] = 1
    coeff_A[0, :, :, 1, 1] = 1
    coeff_B[0, :, :, 0, 0] = 1
    coeff_B[0, :, :, 1, 1] = 1
    
    # Step 1: A1 = A10 + A11, B1 = B00
    coeff_A[1, :, :, 1, 0] = 1
    coeff_A[1, :, :, 1, 1] = 1
    coeff_B[1, :, :, 0, 0] = 1
    
    # Step 2: A2 = A00, B2 = B01 - B11
    coeff_A[2, :, :, 0, 0] = 1
    coeff_B[2, :, :, 0, 1] = 1
    coeff_B[2, :, :, 1, 1] = -1
    
    # Step 3: A3 = A11, B3 = B10 - B00
    coeff_A[3, :, :, 1, 1] = 1
    coeff_B[3, :, :, 1, 0] = 1
    coeff_B[3, :, :, 0, 0] = -1
    
    # Step 4: A4 = A00 + A01, B4 = B11
    coeff_A[4, :, :, 0, 0] = 1
    coeff_A[4, :, :, 0, 1] = 1
    coeff_B[4, :, :, 1, 1] = 1
    
    # Step 5: A5 = A10 - A00, B5 = B00 + B01
    coeff_A[5, :, :, 1, 0] = 1
    coeff_A[5, :, :, 0, 0] = -1
    coeff_B[5, :, :, 0, 0] = 1
    coeff_B[5, :, :, 0, 1] = 1
    
    # Step 6: A6 = A01 - A11, B6 = B10 + B11
    coeff_A[6, :, :, 0, 1] = 1
    coeff_A[6, :, :, 1, 1] = -1
    coeff_B[6, :, :, 1, 0] = 1
    coeff_B[6, :, :, 1, 1] = 1
    
    # Now fill U_4, V_4, W_4
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            # U_4
            for i_global in range(16):
                i_row = i_global // 4
                i_col = i_global % 4
                br = i_row // 2
                lr = i_row % 2
                bc = i_col // 2
                lc = i_col % 2
                # U_4[i_global, col] = sum_{lr', lc'} U_s[lr'*2+lc', t] * coeff_A[s, lr', lc', br, bc]
                # Wait, U_s[lr*2+lc, t] is the coefficient for A_s[lr, lc].
                # A_s[lr, lc] = sum_{br, bc} coeff_A[s, lr, lc, br, bc] * A[br, bc, lr, lc]
                # So U_4[i_global, col] = U_s[lr*2+lc, t] * coeff_A[s, lr, lc, br, bc]
                # Because i_global corresponds to a specific br, bc, lr, lc.
                U_4[i_global, col] = U_s[lr*2+lc, t] * coeff_A[s, lr, lc, br, bc]
                
            # V_4
            for j_global in range(16):
                j_row = j_global // 4
                j_col = j_global % 4
                br = j_row // 2
                lr = j_row % 2
                bc = j_col // 2
                lc = j_col % 2
                V_4[j_global, col] = V_s[lr*2+lc, t] * coeff_B[s, lr, lc, br, bc]
                
            # W_4
            for k_global in range(16):
                k_row = k_global // 4
                k_col = k_global % 4
                br = k_row // 2
                lr = k_row % 2
                bc = k_col // 2
                lc = k_col % 2
                W_4[k_global, col] = W_s[lr*2+lc, t] * 1 # coeff_C is 1 for all?
                # C_s = A_s * B_s
                # C_s[lr, lc] = ...
                # W_s gives the coefficients for C_s.
                # C_s is added to C.
                # The position of C_s in C is determined by s.
                # C00 = M1 + M4 - M5 + M7
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M1 - M2 + M3 + M6
                # So C_s contributes to specific blocks of C.
                # This means W_4 is not just W_s * 1.
                # We need to map the output of C_s to the global C.
                
    # The mapping of C_s to C is:
    # C00 gets M1, M4, -M5, M7
    # C01 gets M3, M5
    # C10 gets M2, M4
    # C11 gets M1, -M2, M3, M6
    
    # So for a given s (which corresponds to M_s), the output C_s is placed in a specific block of C.
    # And within that block, it is placed with a specific sign.
    
    # Let's define the target block and sign for each s.
    target_block = [ (0,0), (1,0), (0,1), (1,1), (0,1), (1,0), (1,1) ] # M1..M7
    # M1: C00 (+), C11 (+)
    # M2: C10 (+), C11 (-)
    # M3: C01 (+), C11 (+)
    # M4: C00 (+), C10 (+)
    # M5: C00 (-), C01 (+)
    # M6: C11 (+)
    # M7: C00 (+)
    
    # Wait, the standard Strassen formulas are:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So:
    # M1 -> C00 (+1), C11 (+1)
    # M2 -> C10 (+1), C11 (-1)
    # M3 -> C01 (+1), C11 (+1)
    # M4 -> C00 (+1), C10 (+1)
    # M5 -> C00 (-1), C01 (+1)
    # M6 -> C11 (+1)
    # M7 -> C00 (+1)
    
    # Let's re-verify the W_4 construction.
    # For each s, t:
    # col = s*7 + t
    # For each k_global (0..15):
    #   k_row, k_col -> br, lr, bc, lc
    #   We need to check if C_s contributes to C[br, bc, lr, lc].
    #   If yes, with what sign?
    
    # Let's create a dictionary or array for the contribution of M_s to each block.
    # contrib[s][br][bc] = sign
    
    contrib = np.zeros((7, 2, 2))
    # M1 (s=0)
    contrib[0, 0, 0] = 1
    contrib[0, 1, 1] = 1
    # M2 (s=1)
    contrib[1, 1, 0] = 1
    contrib[1, 1, 1] = -1
    # M3 (s=2)
    contrib[2, 0, 1] = 1
    contrib[2, 1, 1] = 1
    # M4 (s=3)
    contrib[3, 0, 0] = 1
    contrib[3, 1, 0] = 1
    # M5 (s=4)
    contrib[4, 0, 0] = -1
    contrib[4, 0, 1] = 1
    # M6 (s=5)
    contrib[5, 1, 1] = 1
    # M7 (s=6)
    contrib[6, 0, 0] = 1
    
    # Now fill W_4 correctly
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            for k_global in range(16):
                k_row = k_global // 4
                k_col = k_global % 4
                br = k_row // 2
                lr = k_row % 2
                bc = k_col // 2
                lc = k_col % 2
                W_4[k_global, col] = W_s[lr*2+lc, t] * contrib[s, br, bc]
                
    # Copy U_4, V_4, W_4 to U, V, W
    U[:, :49] = U_4
    # V_4 is 16x49. But V in the main problem is 16x7.
    # Wait, the main problem is 4x4x7.
    # The first part is 4x4x4. So V_4 corresponds to the first 4 columns of B.
    # V_global (16, 49) -> V[:, :49] = V_4
    V[:, :49] = V_4
    W[:, :49] = W_4
    
    # --- Part 2: 4x4x3 Schoolbook ---
    # Rank 48.
    # U_2 (16, 48), V_2 (12, 48), W_2 (12, 48)
    # Note: V and W have 12 rows (4*3) for this part, but the global V, W have 28 rows (4*7).
    # We need to place this in the correct rows.
    # The first 4 columns of B are handled by Part 1.
    # The last 3 columns of B are handled by Part 2.
    # So V_2 should be placed in rows 16..27 of V.
    # W_2 should be placed in rows 16..27 of W.
    # U_2 is the same as U for Part 1, but with different columns.
    # Actually, U is the same for all parts.
    # So U_2 will be placed in columns 49..96 of U.
    
    U_2 = np.zeros((16, 48))
    V_2 = np.zeros((12, 48))
    W_2 = np.zeros((12, 48))
    
    col_offset = 49
    for i in range(4):
        for j in range(4):
            for k in range(3):
                col = col_offset + (i*4 + j)*3 + k
                U_2[i*4 + j, col - col_offset] = 1
                V_2[j*3 + k, col - col_offset] = 1
                W_2[k*4 + i, col - col_offset] = 1
                
    U[:, 49:] = U_2
    V[16:, 49:] = V_2
    W[16:, 49:] = W_2
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
