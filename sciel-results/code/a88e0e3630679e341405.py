"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 4
    
    # We use a block-Strassen approach.
    # A is 2x4, B is 4x4.
    # Split A into columns: A = [A1 A2] where A1 is 2x2, A2 is 2x2.
    # Split B into columns: B = [B1 B2] where B1 is 4x2, B2 is 4x2.
    # This doesn't help directly because A is 2x4.
    
    # Better: Split the middle dimension m=4 into two parts of size 2.
    # A (2x4) = [A_left A_right] where A_left is 2x2, A_right is 2x2.
    # B (4x4) = [B_top; B_bottom] where B_top is 2x4, B_bottom is 2x4.
    # Then C = A * B = A_left * B_top + A_right * B_bottom.
    
    # Now we have two matrix multiplications of type 2x2 * 2x4.
    # Let's analyze 2x2 * 2x4.
    # A' is 2x2, B' is 2x4.
    # Split B' into two 2x2 blocks: B' = [B1 B2].
    # C' = A' * B' = [A'*B1 A'*B2].
    # A'*B1 is 2x2 * 2x2 -> rank 7 (Strassen).
    # A'*B2 is 2x2 * 2x2 -> rank 7 (Strassen).
    # So 2x2 * 2x4 has rank 7 + 7 = 14?
    # Wait, can we do better?
    # The tensor for 2x2 * 2x4 is the sum of two 2x2 * 2x2 tensors.
    # Rank(2x2*2x4) <= Rank(2x2*2x2) + Rank(2x2*2x2) = 7 + 7 = 14.
    # Is it exactly 14?
    # The standard algorithm for 2x2*2x4 would be 2*2*4 = 16.
    # Using Strassen on the 2x2 blocks gives 14.
    
    # So for the original 2x4 * 4x4:
    # C = A_left * B_top + A_right * B_bottom.
    # A_left * B_top is 2x2 * 2x4. Rank <= 14.
    # A_right * B_bottom is 2x2 * 2x4. Rank <= 14.
    # Total rank <= 14 + 14 = 28.
    
    # Can we do better by mixing?
    # The structure is linear.
    # Let's verify if 28 is the minimum.
    # Known bounds for 2x4*4x4:
    # Schoolbook: 32.
    # Block Strassen: 28.
    # It is unlikely to be lower than 28 given the sub-structure.
    
    # Let's construct the 28-rank decomposition.
    
    # Helper: Strassen 2x2 * 2x2 decomposition
    # Inputs: A (2x2), B (2x2)
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map this to the global indices.
    # Global A: 2x4. Indices (i, j) where i in 0..1, j in 0..3.
    # A_left: i in 0..1, j in 0..1. (Global j 0,1)
    # A_right: i in 0..1, j in 2..3. (Global j 2,3)
    
    # Global B: 4x4. Indices (j, k) where j in 0..3, k in 0..3.
    # B_top: j in 0..1, k in 0..3.
    # B_bottom: j in 2..3, k in 0..3.
    
    # C: 2x4. Indices (i, k) where i in 0..1, k in 0..3.
    
    # Decomposition 1: A_left * B_top
    # A_left is 2x2. Let's call its elements a00, a01, a10, a11.
    # Global A indices: (0,0)->a00, (0,1)->a01, (1,0)->a10, (1,1)->a11.
    # B_top is 2x4. Split into B_top_left (2x2) and B_top_right (2x2).
    # B_top_left: Global B (0,0)..(1,1). Indices (0,0)->b00, (0,1)->b01, (1,0)->b10, (1,1)->b11.
    # B_top_right: Global B (0,2)..(1,3). Indices (0,2)->b02, (0,3)->b03, (1,2)->b12, (1,3)->b13.
    
    # Product C_top_left (2x2) = A_left * B_top_left.
    # Product C_top_right (2x2) = A_left * B_top_right.
    
    # Decomposition 2: A_right * B_bottom
    # A_right is 2x2. Global A indices: (0,2)->c00, (0,3)->c01, (1,2)->c10, (1,3)->c11.
    # B_bottom is 2x4. Split into B_bottom_left (2x2) and B_bottom_right (2x2).
    # B_bottom_left: Global B (2,0)..(3,1). Indices (2,0)->d00, (2,1)->d01, (3,0)->d10, (3,1)->d11.
    # B_bottom_right: Global B (2,2)..(3,3). Indices (2,2)->d02, (2,3)->d03, (3,2)->d12, (3,3)->d13.
    
    # Product C_bot_left (2x2) = A_right * B_bottom_left.
    # Product C_bot_right (2x2) = A_right * B_bottom_right.
    
    # Total C = [C_top_left + C_bot_left, C_top_right + C_bot_right].
    
    # We will define 4 Strassen decompositions, each of rank 7.
    # Total rank 28.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Helper to add a Strassen decomposition
    # a_indices: list of 4 indices into U (flattened i*m+j)
    # b_indices: list of 4 indices into V (flattened j*p+k)
    # w_indices: list of 4 indices into W (flattened k*n+i)
    # offset: starting column index in U, V, W
    
    def add_strassen(a_idx, b_idx, w_idx, offset):
        # a_idx: [a00, a01, a10, a11]
        # b_idx: [b00, b01, b10, b11]
        # w_idx: [c00, c01, c10, c11]
        
        # M1 = (A00+A11)*(B00+B11)
        U[a_idx[0]][offset + 0] += 1
        U[a_idx[3]][offset + 0] += 1
        V[b_idx[0]][offset + 0] += 1
        V[b_idx[3]][offset + 0] += 1
        W[w_idx[0]][offset + 0] += 1
        W[w_idx[1]][offset + 0] += 1
        W[w_idx[2]][offset + 0] += 1
        W[w_idx[3]][offset + 0] += 1
        
        # M2 = (A10+A11)*B00
        U[a_idx[2]][offset + 1] += 1
        U[a_idx[3]][offset + 1] += 1
        V[b_idx[0]][offset + 1] += 1
        W[w_idx[2]][offset + 1] += 1
        W[w_idx[3]][offset + 1] -= 1
        
        # M3 = A00*(B01-B11)
        U[a_idx[0]][offset + 2] += 1
        V[b_idx[1]][offset + 2] += 1
        V[b_idx[3]][offset + 2] -= 1
        W[w_idx[1]][offset + 2] += 1
        W[w_idx[3]][offset + 2] += 1
        
        # M4 = A11*(B10-B00)
        U[a_idx[3]][offset + 3] += 1
        V[b_idx[2]][offset + 3] += 1
        V[b_idx[0]][offset + 3] -= 1
        W[w_idx[0]][offset + 3] += 1
        W[w_idx[2]][offset + 3] += 1
        
        # M5 = (A00+A01)*B11
        U[a_idx[0]][offset + 4] += 1
        U[a_idx[1]][offset + 4] += 1
        V[b_idx[3]][offset + 4] += 1
        W[w_idx[0]][offset + 4] -= 1
        W[w_idx[1]][offset + 4] += 1
        
        # M6 = (A10-A00)*(B00+B01)
        U[a_idx[2]][offset + 5] += 1
        U[a_idx[0]][offset + 5] -= 1
        V[b_idx[0]][offset + 5] += 1
        V[b_idx[1]][offset + 5] += 1
        W[w_idx[3]][offset + 5] += 1
        
        # M7 = (A01-A11)*(B10+B11)
        U[a_idx[1]][offset + 6] += 1
        U[a_idx[3]][offset + 6] -= 1
        V[b_idx[2]][offset + 6] += 1
        V[b_idx[3]][offset + 6] += 1
        W[w_idx[0]][offset + 6] += 1
        W[w_idx[1]][offset + 6] -= 1

    # Define indices for the 4 sub-problems
    
    # 1. A_left * B_top_left
    # A_left: (0,0), (0,1), (1,0), (1,1) -> U indices 0, 1, 4, 5
    a1 = [0, 1, 4, 5]
    # B_top_left: (0,0), (0,1), (1,0), (1,1) -> V indices 0, 1, 4, 5
    b1 = [0, 1, 4, 5]
    # C_top_left: (0,0), (0,1), (1,0), (1,1) -> W indices 0, 1, 2, 3
    w1 = [0, 1, 2, 3]
    add_strassen(a1, b1, w1, 0)
    
    # 2. A_left * B_top_right
    # A_left: same as above
    a2 = [0, 1, 4, 5]
    # B_top_right: (0,2), (0,3), (1,2), (1,3) -> V indices 2, 3, 6, 7
    b2 = [2, 3, 6, 7]
    # C_top_right: (0,2), (0,3), (1,2), (1,3) -> W indices 4, 5, 6, 7
    w2 = [4, 5, 6, 7]
    add_strassen(a2, b2, w2, 7)
    
    # 3. A_right * B_bottom_left
    # A_right: (0,2), (0,3), (1,2), (1,3) -> U indices 2, 3, 6, 7
    a3 = [2, 3, 6, 7]
    # B_bottom_left: (2,0), (2,1), (3,0), (3,1) -> V indices 8, 9, 12, 13
    b3 = [8, 9, 12, 13]
    # C_bot_left: (0,0), (0,1), (1,0), (1,1) -> W indices 0, 1, 2, 3
    w3 = [0, 1, 2, 3]
    add_strassen(a3, b3, w3, 14)
    
    # 4. A_right * B_bottom_right
    # A_right: same as above
    a4 = [2, 3, 6, 7]
    # B_bottom_right: (2,2), (2,3), (3,2), (3,3) -> V indices 10, 11, 14, 15
    b4 = [10, 11, 14, 15]
    # C_bot_right: (0,2), (0,3), (1,2), (1,3) -> W indices 4, 5, 6, 7
    w4 = [4, 5, 6, 7]
    add_strassen(a4, b4, w4, 21)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
