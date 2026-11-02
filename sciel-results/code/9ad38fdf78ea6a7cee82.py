import numpy as np

def construct():
    n, m, p = 4, 4, 7
    # Total rank will be 49 + 28 + 16 = 93
    rank = 93
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2x2 decomposition
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W2 = [
        [1, 0, 0, 0, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 1]
    ]
    
    # Schoolbook 2x2x1 decomposition
    # A (2x2) * B (2x1) -> C (2x1)
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # Rank 4
    U21 = [
        [1, 0, 0, 0], # A00
        [0, 1, 0, 0], # A01
        [0, 0, 1, 0], # A10
        [0, 0, 0, 1]  # A11
    ]
    V21 = [
        [1, 0, 1, 0], # B0
        [0, 1, 0, 1]  # B1
    ]
    W21 = [
        [1, 0, 0, 0], # C0
        [0, 0, 1, 0]  # C1
    ]
    # Wait, W21 indices: C0 is row 0, C1 is row 1.
    # Term 1: A00 B0 -> C0. U[0]=1, V[0]=1, W[0]=1.
    # Term 2: A01 B1 -> C0. U[1]=1, V[1]=1, W[0]=1.
    # Term 3: A10 B0 -> C1. U[2]=1, V[0]=1, W[1]=1.
    # Term 4: A11 B1 -> C1. U[3]=1, V[1]=1, W[1]=1.
    # So:
    # U21[0] = [1, 0, 0, 0]
    # U21[1] = [0, 1, 0, 0]
    # U21[2] = [0, 0, 1, 0]
    # U21[3] = [0, 0, 0, 1]
    # V21[0] = [1, 0, 1, 0]
    # V21[1] = [0, 1, 0, 1]
    # W21[0] = [1, 1, 0, 0]
    # W21[1] = [0, 0, 1, 1]

    # Helper to add a block decomposition
    # a_block: (r, c) index in 2x2 block grid for A
    # b_block: (r, c) index in 2x2 block grid for B? 
    # For 4x4x4: A is 2x2 blocks of 2x2. B is 2x2 blocks of 2x2.
    # For 4x4x2: A is 2x2 blocks of 2x2. B is 2x1 blocks of 2x1? 
    # Let's define indices carefully.
    
    # Global indices:
    # A: 4x4. Index (i, j) -> i*4+j.
    # Blocks: 
    # A00: i in 0,1; j in 0,1.
    # A01: i in 0,1; j in 2,3.
    # A10: i in 2,3; j in 0,1.
    # A11: i in 2,3; j in 2,3.
    
    # B: 4x7.
    # For the 4x4x4 part, B is 4x4.
    # B00: i in 0,1; j in 0,1.
    # B01: i in 0,1; j in 2,3.
    # B10: i in 2,3; j in 0,1.
    # B11: i in 2,3; j in 2,3.
    
    # C: 4x4.
    # C00: i in 0,1; j in 0,1.
    # C01: i in 0,1; j in 2,3.
    # C10: i in 2,3; j in 0,1.
    # C11: i in 2,3; j in 2,3.

    # For 4x4x2 part, B is 4x2.
    # B_top: i in 0,1; j in 0,1. (2x2)
    # B_bot: i in 2,3; j in 0,1. (2x2)
    # C is 4x2.
    # C_top: i in 0,1; j in 0,1.
    # C_bot: i in 2,3; j in 0,1.
    
    # For 4x4x1 part, B is 4x1.
    # B_top: i in 0,1; j in 0. (2x1)
    # B_bot: i in 2,3; j in 0. (2x1)
    # C is 4x1.
    # C_top: i in 0,1; j in 0.
    # C_bot: i in 2,3; j in 0.

    col_offset = 0
    
    # Part 1: 4x4x4 (Rank 49)
    # Kronecker product of Strassen with Strassen.
    # We iterate over the 7 Strassen terms for the "outer" 2x2x2,
    # and for each, iterate over the 7 Strassen terms for the "inner" 2x2x2.
    # Actually, it's simpler to just say:
    # For each of the 7 outer Strassen multiplications:
    #   We have a 2x2 block of A, a 2x2 block of B, and a 2x2 block of C.
    #   We perform a 2x2x2 Strassen multiplication on these blocks.
    #   This adds 7 rank-1 tensors for each outer term.
    #   Total 7 * 7 = 49.
    
    # Strassen terms for 2x2x2:
    # 1: (A00+A11)(B00+B11) -> C00, C11
    # 2: (A10+A11)B00 -> C10
    # 3: A00(B01-B11) -> C01
    # 4: A11(B10-B00) -> C10
    # 5: (A00+A01)B11 -> C00, C01
    # 6: (A10-A00)(B00+B01) -> C11
    # 7: (A01-A11)(B10+B11) -> C00, C11
    
    # We need to map the "block" indices to global indices.
    # For 4x4x4:
    # A blocks: 00, 01, 10, 11.
    # B blocks: 00, 01, 10, 11.
    # C blocks: 00, 01, 10, 11.
    
    # Let's define the 7 Strassen terms as lists of (a_block, b_block, c_blocks, signs)
    # But the "blocks" here are 2x2 matrices.
    # So for each term, we need to apply the 2x2x2 Strassen decomposition to the 2x2 matrices.
    
    # This is getting complex to hardcode.
    # Let's use the property that the rank-49 decomposition is the Kronecker product.
    # We can generate the 49 tensors by iterating i in 0..6, j in 0..6.
    # U[i,j] = U2[i] * U2[j] (element-wise? No, Kronecker).
    # Actually, the tensor is $T = T_2 \otimes T_2$.
    # The rank-1 terms are $u_i \otimes u_j$, $v_i \otimes v_j$, $w_i \otimes w_j$.
    
    # Let's implement a generic Kronecker product of two decompositions.
    # Decomp 1: U1, V1, W1 (for 2x2x2)
    # Decomp 2: U2, V2, W2 (for 2x2x2)
    # Result: U, V, W for 4x4x4.
    # U has shape 4x4 x 49.
    # V has shape 4x4 x 49.
    # W has shape 4x4 x 49.
    
    # We will fill U, V, W for the first 49 columns.
    
    # Map 2x2 index (r, c) to 4x4 index.
    # Block (br, bc) of size 2x2.
    # Global row = br*2 + r.
    # Global col = bc*2 + c.
    
    def get_4x4_idx(block_r, block_c, local_r, local_c):
        return (block_r * 2 + local_r) * 4 + (block_c * 2 + local_c)
    
    # Strassen terms for 2x2x2 (indices 0..6)
    # We use the U2, V2, W2 defined above.
    # U2[i][j] is the coefficient for A_{block_i, local_j}?
    # No, U2 is 4x7. Rows are A elements (0..3), Cols are terms (0..6).
    # A elements: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1).
    
    # For the 4x4x4 case:
    # A is 4x4. We view it as 2x2 blocks of 2x2.
    # Let A_{B, L} where B is block index (0..3) and L is local index (0..3).
    # A_{B, L} corresponds to global index get_4x4_idx(B//2, B%2, L//2, L%2).
    
    # The Kronecker product decomposition:
    # Term k (0..48). k = i * 7 + j.
    # U[global_a_idx][k] = U2[local_a1_idx][i] * U2[local_a2_idx][j]
    # where global_a_idx corresponds to block (b1, b2) and local (l1, l2).
    # Actually, A is a matrix. The Kronecker product of two 2x2 matrices A1, A2 gives a 4x4 matrix.
    # A = A1 \otimes A2.
    # Then A \otimes A is not what we want. We want to multiply A (4x4) by B (4x4).
    # If A = A1 \otimes A2 and B = B1 \otimes B2, then AB = (A1 B1) \otimes (A2 B2).
    # So we can use the decomposition of A1 B1 (rank 7) and A2 B2 (rank 7).
    # The total rank is 7*7=49.
    # The terms are combinations of terms from A1 B1 and A2 B2.
    
    # So, for i in 0..6 (terms of A1 B1), j in 0..6 (terms of A2 B2):
    #   k = i * 7 + j
    #   For each element in U2[i], it applies to A1.
    #   For each element in U2[j], it applies to A2.
    #   The global U entry is the product.
    
    # Let's do this.
    for i in range(7):
        for j in range(7):
            k = i * 7 + j
            # U2 is 4x7. U2[a_local][i] is coeff for A1 element a_local.
            # U2[b_local][j] is coeff for A2 element b_local.
            # Global A index: block (br1, bc1) for A1, block (br2, bc2) for A2.
            # Wait, A is 4x4. A1 is 2x2, A2 is 2x2.
            # A = A1 \otimes A2.
            # Element (r1, c1) of A1 and (r2, c2) of A2 correspond to global (r1*2+r2, c1*2+c2).
            # So we iterate over a1_local (0..3) and a2_local (0..3).
            # a1_local corresponds to (r1, c1) in A1.
            # a2_local corresponds to (r2, c2) in A2.
            # Global row = r1*2 + r2.
            # Global col = c1*2 + c2.
            
            # U2[a1_local][i] is the value for A1.
            # U2[a2_local][j] is the value for A2.
            
            for a1_l in range(4):
                val1 = U2[a1_l][i]
                if val1 == 0: continue
                r1, c1 = divmod(a1_l, 2)
                for a2_l in range(4):
                    val2 = U2[a2_l][j]
                    if val2 == 0: continue
                    r2, c2 = divmod(a2_l, 2)
                    global_row = r1 * 2 + r2
                    global_col = c1 * 2 + c2
                    global_idx = global_row * 4 + global_col
                    U[global_idx][k] += val1 * val2
            
            # Similarly for V and W.
            # V is for B. B = B1 \otimes B2.
            for b1_l in range(4):
                val1 = V2[b1_l][i]
                if val1 == 0: continue
                r1, c1 = divmod(b1_l, 2)
                for b2_l in range(4):
                    val2 = V2[b2_l][j]
                    if val2 == 0: continue
                    r2, c2 = divmod(b2_l, 2)
                    global_row = r1 * 2 + r2
                    global_col = c1 * 2 + c2
                    global_idx = global_row * 4 + global_col
                    V[global_idx][k] += val1 * val2
            
            for c1_l in range(4):
                val1 = W2[c1_l][i]
                if val1 == 0: continue
                r1, c1 = divmod(c1_l, 2)
                for c2_l in range(4):
                    val2 = W2[c2_l][j]
                    if val2 == 0: continue
                    r2, c2 = divmod(c2_l, 2)
                    global_row = r1 * 2 + r2
                    global_col = c1 * 2 + c2
                    global_idx = global_row * 4 + global_col
                    W[global_idx][k] += val1 * val2
    
    col_offset += 49
    
    # Part 2: 4x4x2 (Rank 28)
    # B is 4x2. We view B as B1 \otimes B2?
    # B1 is 2x2, B2 is 2x1?
    # If B = B1 \otimes B2, then B is 4x2.
    # B1 is 2x2, B2 is 2x1.
    # Then A B = (A1 B1) \otimes (A2 B2).
    # A1 B1 is 2x2x2 (Rank 7).
    # A2 B2 is 2x2x1 (Rank 4).
    # Total rank 7 * 4 = 28.
    
    # We need U21, V21, W21 for 2x2x1.
    # U21 (4x4), V21 (2x4), W21 (2x4).
    
    # Let's define U21, V21, W21 correctly.
    # A (2x2) * B (2x1) -> C (2x1)
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # Terms:
    # 1: A00 B0 -> C0
    # 2: A01 B1 -> C0
    # 3: A10 B0 -> C1
    # 4: A11 B1 -> C1
    
    U21 = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    V21 = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    W21 = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    for i in range(7): # Strassen term for A1 B1
        for j in range(4): # Schoolbook term for A2 B2
            k = col_offset + i * 4 + j
            # U: A = A1 \otimes A2
            for a1_l in range(4):
                val1 = U2[a1_l][i]
                if val1 == 0: continue
                r1, c1 = divmod(a1_l, 2)
                for a2_l in range(4):
                    val2 = U21[a2_l][j]
                    if val2 == 0: continue
                    r2, c2 = divmod(a2_l, 2)
                    global_row = r1 * 2 + r2
                    global_col = c1 * 2 + c2
                    global_idx = global_row * 4 + global_col
                    U[global_idx][k] += val1 * val2
            
            # V: B = B1 \otimes B2
            # B1 is 2x2, B2 is 2x1.
            # B1 local index (r1, c1). B2 local index (r2, c2=0).
            # Global row = r1*2 + r2.
            # Global col = c1*1 + c2? No, B is 4x2.
            # B = B1 \otimes B2.
            # B1 (2x2) \otimes B2 (2x1) -> (4x2).
            # Element (r1, c1) of B1 and (r2, c2) of B2 -> global (r1*2+r2, c1*1+c2).
            # c2 is always 0.
            for b1_l in range(4):
                val1 = V2[b1_l][i]
                if val1 == 0: continue
                r1, c1 = divmod(b1_l, 2)
                for b2_l in range(2):
                    val2 = V21[b2_l][j]
                    if val2 == 0: continue
                    r2, c2 = divmod(b2_l, 1) # c2 is 0
                    global_row = r1 * 2 + r2
                    global_col = c1 * 1 + c2
                    global_idx = global_row * 4 + global_col
                    V[global_idx][k] += val1 * val2
            
            # W: C = C1 \otimes C2
            # C1 is 2x2, C2 is 2x1.
            for c1_l in range(4):
                val1 = W2[c1_l][i]
                if val1 == 0: continue
                r1, c1 = divmod(c1_l, 2)
                for c2_l in range(2):
                    val2 = W21[c2_l][j]
                    if val2 == 0: continue
                    r2, c2 = divmod(c2_l, 1)
                    global_row = r1 * 2 + r2
                    global_col = c1 * 1 + c2
                    global_idx = global_row * 4 + global_col
                    W[global_idx][k] += val1 * val2
    
    col_offset += 28
    
    # Part 3: 4x4x1 (Rank 16)
    # B is 4x1. B = B1 \otimes B2.
    # B1 is 2x2, B2 is 2x1? No, B is 4x1.
    # If B1 is 2x1 and B2 is 2x1, then B1 \otimes B2 is 4x1.
    # A is 4x4. A = A1 \otimes A2.
    # A1 is 2x2, A2 is 2x2.
    # A B = (A1 B1) \otimes (A2 B2)?
    # A1 (2x2) * B1 (2x1) -> 2x1.
    # A2 (2x2) * B2 (2x1) -> 2x1.
    # Result is 4x1.
    # Rank(A1 B1) = 4. Rank(A2 B2) = 4.
    # Total rank 16.
    
    for i in range(4): # Schoolbook term for A1 B1
        for j in range(4): # Schoolbook term for A2 B2
            k = col_offset + i * 4 + j
            # U: A = A1 \otimes A2
            for a1_l in range(4):
                val1 = U21[a1_l][i]
                if val1 == 0: continue
                r1, c1 = divmod(a1_l, 2)
                for a2_l in range(4):
                    val2 = U21[a2_l][j]
                    if val2 == 0: continue
                    r2, c2 = divmod(a2_l, 2)
                    global_row = r1 * 2 + r2
                    global_col = c1 * 2 + c2
                    global_idx = global_row * 4 + global_col
                    U[global_idx][k] += val1 * val2
            
            # V: B = B1 \otimes B2
            # B1 (2x1) \otimes B2 (2x1) -> 4x1.
            # Global row = r1*2 + r2.
            # Global col = 0.
            for b1_l in range(2):
                val1 = V21[b1_l][i]
                if val1 == 0: continue
                r1, c1 = divmod(b1_l, 1)
                for b2_l in range(2):
                    val2 = V21[b2_l][j]
                    if val2 == 0: continue
                    r2, c2 = divmod(b2_l, 1)
                    global_row = r1 * 2 + r2
                    global_col = 0
                    global_idx = global_row * 4 + global_col
                    V[global_idx][k] += val1 * val2
            
            # W: C = C1 \otimes C2
            # C1 (2x1) \otimes C2 (2x1) -> 4x1.
            for c1_l in range(2):
                val1 = W21[c1_l][i]
                if val1 == 0: continue
                r1, c1 = divmod(c1_l, 1)
                for c2_l in range(2):
                    val2 = W21[c2_l][j]
                    if val2 == 0: continue
                    r2, c2 = divmod(c2_l, 1)
                    global_row = r1 * 2 + r2
                    global_col = 0
                    global_idx = global_row * 4 + global_col
                    W[global_idx][k] += val1 * val2
    
    return {"U": U, "V": V, "W": W}
