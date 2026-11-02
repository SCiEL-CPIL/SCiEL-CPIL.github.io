"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strategy:
    # 1. Split B (4x6) into B_left (4x4) and B_right (4x2).
    # 2. Compute C_left = A * B_left (4x4) using Strassen (Rank 49).
    # 3. Compute C_right = A * B_right (4x2) using Schoolbook (Rank 32).
    # 4. Total Rank = 49 + 32 = 81.
    
    rank_strassen_4x4x4 = 49
    rank_schoolbook_4x4x2 = 32
    total_rank = rank_strassen_4x4x4 + rank_schoolbook_4x4x2
    
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 81
    V = [[0] * total_rank for _ in range(m * p)]  # 24 x 81
    W = [[0] * total_rank for _ in range(p * n)]  # 24 x 81
    
    # --- Part 1: Strassen for 4x4 x 4x4 ---
    # We need to construct the 49-rank decomposition.
    # This is a composition of 7 Strassen steps, each using 7 rank-1 terms for 2x2x2.
    
    # Helper: Get the 2x2 Strassen decomposition coefficients
    # Returns U_2x2 (4x7), V_2x2 (4x7), W_2x2 (4x7)
    def get_2x2_strassen():
        U2 = [[0]*7 for _ in range(4)]
        V2 = [[0]*7 for _ in range(4)]
        W2 = [[0]*7 for _ in range(4)]
        
        # Indices for 2x2 matrix: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # A indices: 0:A11, 1:A12, 2:A21, 3:A22
        # B indices: 0:B11, 1:B12, 2:B21, 3:B22
        # C indices: 0:C11, 1:C12, 2:C21, 3:C22
        
        # M1 = (A11+A22)(B11+B22)
        # U2: A11(0), A22(3)
        U2[0][0] = 1; U2[3][0] = 1
        # V2: B11(0), B22(3)
        V2[0][0] = 1; V2[3][0] = 1
        # W2: C11(0), C22(3)
        W2[0][0] = 1; W2[3][0] = 1
        
        # M2 = (A21+A22)B11
        U2[2][1] = 1; U2[3][1] = 1
        V2[0][1] = 1
        W2[2][1] = 1 # C21 += M2
        
        # M3 = A11(B12-B22)
        U2[0][2] = 1
        V2[1][2] = 1; V2[3][2] = -1
        W2[1][2] = 1 # C12 += M3
        
        # M4 = A22(B21-B11)
        U2[3][3] = 1
        V2[2][3] = 1; V2[0][3] = -1
        W2[0][3] = -1 # C11 -= M4
        W2[2][3] = 1  # C21 += M4
        
        # M5 = (A11+A12)B22
        U2[0][4] = 1; U2[1][4] = 1
        V2[3][4] = 1
        W2[1][4] = 1 # C12 += M5
        W2[3][4] = -1 # C22 -= M5
        
        # M6 = (A21-A11)(B11+B12)
        U2[2][5] = 1; U2[0][5] = -1
        V2[0][5] = 1; V2[1][5] = 1
        W2[0][5] = 1 # C11 += M6
        
        # M7 = (A12-A22)(B21+B22)
        U2[1][6] = 1; U2[3][6] = -1
        V2[2][6] = 1; V2[3][6] = 1
        W2[0][6] = 1 # C11 += M7
        W2[3][6] = 1 # C22 += M7
        
        return U2, V2, W2

    U2, V2, W2 = get_2x2_strassen()
    
    # Define the 7 Strassen combinations for 4x4x4
    # Each step s has A_comb (2x2 of blocks) and B_comb (2x2 of blocks)
    # A blocks: A00, A01, A10, A11 (2x2 each)
    # B blocks: B00, B01, B10, B11 (2x2 each)
    
    # Step 0: M1 = (A00+A11)(B00+B11)
    # Step 1: M2 = (A10+A11)B00
    # Step 2: M3 = A00(B01-B11)
    # Step 3: M4 = A11(B10-B00)
    # Step 4: M5 = (A00+A01)B11
    # Step 5: M6 = (A10-A00)(B00+B01)
    # Step 6: M7 = (A01-A11)(B10+B11)
    
    # We map block indices to linear indices in U (16x) and V (16x for 4x4 B)
    # A linear index: i*4 + j
    # Block (br, bc) covers rows 2*br..2*br+1, cols 2*bc..2*bc+1
    # A block (br, bc) linear indices:
    # (2*br)*4 + 2*bc, (2*br)*4 + 2*bc+1, (2*br+1)*4 + 2*bc, (2*br+1)*4 + 2*bc+1
    
    # B block (br, bc) linear indices (for 4x4 B):
    # (2*br)*4 + 2*bc, ... same pattern
    
    # For each Strassen step s, we add 7 columns to U, V, W.
    for s in range(7):
        # Determine A_comb and B_comb block coefficients
        # a_coeff[br][bc] for A
        # b_coeff[br][bc] for B
        a_coeff = [[0,0],[0,0]]
        b_coeff = [[0,0],[0,0]]
        
        if s == 0:
            a_coeff[0][0] = 1; a_coeff[1][1] = 1
            b_coeff[0][0] = 1; b_coeff[1][1] = 1
        elif s == 1:
            a_coeff[1][0] = 1; a_coeff[1][1] = 1
            b_coeff[0][0] = 1
        elif s == 2:
            a_coeff[0][0] = 1
            b_coeff[0][1] = 1; b_coeff[1][1] = -1
        elif s == 3:
            a_coeff[1][1] = 1
            b_coeff[1][0] = 1; b_coeff[0][0] = -1
        elif s == 4:
            a_coeff[0][0] = 1; a_coeff[0][1] = 1
            b_coeff[1][1] = 1
        elif s == 5:
            a_coeff[1][0] = 1; a_coeff[0][0] = -1
            b_coeff[0][0] = 1; b_coeff[0][1] = 1
        elif s == 6:
            a_coeff[0][1] = 1; a_coeff[1][1] = -1
            b_coeff[1][0] = 1; b_coeff[1][1] = 1
            
        # Now apply 2x2 Strassen to the combined blocks
        for t in range(7):
            col = s * 7 + t
            
            # U update
            # U2[t][idx] gives coeff for A_block_local_idx
            # We need to map local block idx to global A block
            # Local 2x2 matrix indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            for local_idx in range(4):
                val = U2[t][local_idx]
                if val != 0:
                    # Determine which A block this local_idx corresponds to
                    # local_idx 0 -> block (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1)
                    lr = local_idx // 2
                    lc = local_idx % 2
                    coeff = a_coeff[lr][lc]
                    if coeff != 0:
                        # Add coeff * val to U for all elements in A block (lr, lc)
                        for dr in range(2):
                            for dc in range(2):
                                i = 2*lr + dr
                                j = 2*lc + dc
                                U[i*m + j][col] += coeff * val
                                
            # V update
            for local_idx in range(4):
                val = V2[t][local_idx]
                if val != 0:
                    lr = local_idx // 2
                    lc = local_idx % 2
                    coeff = b_coeff[lr][lc]
                    if coeff != 0:
                        # Add coeff * val to V for all elements in B block (lr, lc)
                        # B is 4x4 here, so indices are j*4 + k
                        for dr in range(2):
                            for dc in range(2):
                                j = 2*lr + dr
                                k = 2*lc + dc
                                V[j*p + k][col] += coeff * val
                                
            # W update
            # W2[t][idx] gives coeff for C_block_local_idx
            # C is 4x4. Output of this 4x4x4 part is C_left.
            # C_left indices: i*4 + k
            for local_idx in range(4):
                val = W2[t][local_idx]
                if val != 0:
                    lr = local_idx // 2
                    lc = local_idx % 2
                    # The result of the 2x2 block multiplication is added to C block (lr, lc)
                    # So we add val to W for all elements in C block (lr, lc)
                    for dr in range(2):
                        for dc in range(2):
                            i = 2*lr + dr
                            k = 2*lc + dc
                            W[k*n + i][col] += val

    # --- Part 2: Schoolbook for 4x4 x 4x2 ---
    # Rank 32
    col_offset = rank_strassen_4x4x4
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(2): # k_local 0..1
                # Global k is 4 + k_local
                k_global = 4 + k
                U[i*m + j][col_offset + col_idx] = 1
                V[j*p + k_global][col_offset + col_idx] = 1
                W[k_global*n + i][col_offset + col_idx] = 1
                col_idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
