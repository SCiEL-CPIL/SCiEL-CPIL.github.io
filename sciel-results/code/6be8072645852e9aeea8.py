"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _schoolbook_block(n, m, p, offset, U, V, W):
    """Add schoolbook rank n*m*p block at given offset into U,V,W."""
    col = offset
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i][col] = 1
                V[j][col] = 1
                W[k][col] = 1
                col += 1
    return col


def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # Strategy: split the 5x5x5 tensor into a 4x4x4 block (rank 64 schoolbook) 
    # plus the remaining "fringe" blocks.
    # The fringe consists of:
    # - Top row of C (i=4, j=0..4, k=0..4): depends on A row 4 and B cols 0..4
    #   Actually, let's be precise about the tensor indices.
    # Tensor T[i,j,k] is 1 if and only if the multiplication C[i,k] = sum_j A[i,j] B[j,k] 
    # includes the term A[i,j]B[j,k].
    # So T[i,j,k] = 1 for all i,j,k.
    # 
    # We split indices into {0..3} and {4}.
    # Block 1: i in 0..3, j in 0..3, k in 0..3. Rank 64 (4*4*4).
    # The rest is a union of blocks where at least one index is 4.
    # Let's compute the rank of the "rest" using schoolbook for each sub-block.
    # 
    # The indices are (i, j, k).
    # Case 1: i=4, j=4, k=4 -> 1 tensor (1,1,1). Rank 1.
    # Case 2: i=4, j=4, k in 0..3 -> 4 tensors (1,1,4). Rank 4.
    # Case 3: i=4, j in 0..3, k=4 -> 4 tensors (1,4,1). Rank 4.
    # Case 4: i in 0..3, j=4, k=4 -> 4 tensors (4,1,1). Rank 4.
    # Case 5: i=4, j in 0..3, k in 0..3 -> 16 tensors (1,4,4). Rank 16.
    # Case 6: i in 0..3, j=4, k in 0..3 -> 16 tensors (4,1,4). Rank 16.
    # Case 7: i in 0..3, j in 0..3, k=4 -> 16 tensors (4,4,1). Rank 16.
    #
    # Total rank = 64 (core) + 1 + 4 + 4 + 4 + 16 + 16 + 16 = 125. 
    # Wait, this is just schoolbook again! 64 + 61 = 125.
    # Why? Because schoolbook rank for (a,b,c) is a*b*c.
    # Sum of a*b*c over all combinations of {0..3} and {4} is exactly (4+1)*(4+1)*(4+1) = 125.
    # So naive block splitting doesn't help if we use schoolbook for each block.
    #
    # We need to use a lower-rank decomposition for the blocks, OR exploit symmetries.
    #
    # Known lower bounds and ranks:
    # 2x2x2: rank 7 (Strassen).
    # 3x3x3: rank 23? (Laderman is 23 for 3x3).
    # 4x4x4: rank 49 (Winograd).
    #
    # Let's try Strassen for 2x2x2 blocks.
    # Split 5 into 2+2+1.
    # This creates 3x3x3 = 27 blocks.
    # This is too complex to hardcode without errors.
    #
    # Alternative: Use the known Strassen algorithm for 2x2x2.
    # Let's try to use Strassen for a 2x2x2 sub-problem within the 5x5x5.
    # But we must cover the whole 5x5x5.
    #
    # Let's look at the "fringe" again.
    # The core 4x4x4 has rank 64 with schoolbook.
    # Can we do better than 64 for 4x4x4? Yes, Winograd rank 49.
    # If we use Winograd for 4x4x4, we save 64-49=15 rank.
    # New rank: 49 + 61 = 110.
    # Score: 125/110 = 1.136.
    #
    # Can we do better for the fringe?
    # The fringe blocks are:
    # (1,4,4), (4,1,4), (4,4,1), (1,1,4), (1,4,1), (4,1,1), (1,1,1).
    # Ranks: 16, 16, 16, 4, 4, 4, 1. Total 61.
    #
    # Is there a way to reduce the fringe rank?
    # Consider (1,4,4). This is C[4, k] = sum_j A[4, j] B[j, k] for k=0..3.
    # This is a matrix-vector multiplication of size 4x4.
    # It's 4 dot products of length 4.
    # Can we do it with less than 16 mults?
    # No, matrix-vector mult of size n has rank n (for the output) but here we have 4 outputs.
    # Actually, (1, 4, 4) tensor: U is 4xR, V is 16xR, W is 4xR.
    # Wait, U size is i*m = 1*4 = 4. V size is m*p = 4*4 = 16. W size is p*n = 4*1 = 4.
    # The tensor is T[i, j, k] where i=0 (since i in 0..0), j in 0..3, k in 0..3.
    # T[0, j, k] = A[4, j] * B[j, k].
    # This is just the product of a row vector and a matrix.
    # It can be done in 4*4=16 mults.
    # Is rank < 16 possible?
    # The tensor is sum_j A[4,j] e_j \otimes B[j,:] \otimes e_0? No.
    # It's sum_{j,k} A[4,j] B[j,k] e_0 \otimes e_j \otimes e_k.
    # This is a rank-1 tensor in the first dimension, but full rank in the others.
    # Actually, it's just 16 independent products if A and B are arbitrary.
    # So rank is 16.
    #
    # So the only saving is in the 4x4x4 core.
    # Winograd 4x4x4 rank 49.
    # Total rank 110.
    #
    # Let's implement Winograd 4x4x4.
    # This is complex. I will use a known factorization.
    # Alternatively, I can use Strassen recursively?
    # 4x4x4 can be done by 2x2x2 Strassen on 2x2 blocks?
    # 4x4 = [2x2 2x2; 2x2 2x2].
    # This leads to 7 * (2x2x2) operations? No, it leads to 7 * (2x2 matrix mults).
    # Each 2x2 matrix mult has rank 7.
    # So 7 * 7 = 49.
    # Yes! Strassen for 4x4x4 has rank 49.
    #
    # So I can implement Strassen for 4x4x4 by applying Strassen to the 2x2 blocks.
    # But I need to construct the U, V, W factors explicitly.
    #
    # Let's write a helper to build Strassen 2x2x2 factors.
    # Then combine them for 4x4x4.
    
    # Strassen 2x2x2 factors:
    # U (2*2, 7) = (4, 7)
    # V (2*2, 7) = (4, 7)
    # W (2*2, 7) = (4, 7)
    
    # Indices for Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # U rows: A00, A01, A10, A11
    # V rows: B00, B01, B10, B11
    # W rows: C00, C01, C10, C11
    
    # U:
    # M1: A00+1, A11+1
    # M2: A10+1, A11+1
    # M3: A00+1
    # M4: A11+1
    # M5: A00+1, A01+1
    # M6: A10+1, A00-1
    # M7: A01+1, A11-1
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ]
    
    # V:
    # M1: B00+1, B11+1
    # M2: B00+1
    # M3: B10+1, B11-1
    # M4: B01+1, B10-1
    # M5: B11+1
    # M6: B00+1, B10+1
    # M7: B01+1, B11+1
    
    V2 = [
        [1, 1, 0, 0, 0, 1, 0],  # B00
        [0, 0, 0, 1, 0, 0, 1],  # B01
        [0, 0, 1, -1, 0, 1, 0], # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    
    # W:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    W2 = [
        [1, 0, 0, -1, 1, 0, -1],  # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0]    # C11
    ]
    
    # Now build 4x4x4 Strassen factors using 2x2 blocks.
    # The 4x4 matrix is divided into 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where each Ai,j is 2x2.
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # We apply Strassen to the 2x2 block matrix.
    # The "elements" of the block matrix are 2x2 matrices.
    # So the multiplications M1...M7 are products of 2x2 matrices.
    # Each 2x2 multiplication is done with Strassen rank 7.
    
    # Total rank 7*7 = 49.
    
    # We need to construct U4, V4, W4 of size (16, 49), (16, 49), (16, 49).
    # U4 indices: i in 0..3, j in 0..3. Flattened as i*4+j.
    # V4 indices: j in 0..3, k in 0..3. Flattened as j*4+k.
    # W4 indices: k in 0..3, i in 0..3. Flattened as k*4+i.
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    # Strassen block indices (r, c) for 2x2 block matrix.
    # Block (0,0) corresponds to rows 0,1 and cols 0,1 of the 4x4 matrix.
    # So element (0,0) of block matrix is A[0..1, 0..1].
    # Flattened index in U4 for A[i,j] is i*4+j.
    # For block (br, bc), the rows are 2*br, 2*br+1. Cols are 2*bc, 2*bc+1.
    # So the 4 indices in U4 for block (br, bc) are:
    # (2*br)*4 + 2*bc, (2*br)*4 + 2*bc+1, (2*br+1)*4 + 2*bc, (2*br+1)*4 + 2*bc+1.
    
    for br in range(2):
        for bc in range(2):
            # Indices in U4 for this block
            u_idx = [
                (2*br)*4 + 2*bc,
                (2*br)*4 + 2*bc + 1,
                (2*br+1)*4 + 2*bc,
                (2*br+1)*4 + 2*bc + 1
            ]
            
            # Indices in V4 for this block (same bc for cols, but rows are bc? No.)
            # Wait, V4 indices are j*4+k.
            # The "rows" of B in block (br, bc) are the "cols" of A block (br, bc).
            # So the j indices for B block (br, bc) are the same as i indices for A block (br, bc)?
            # No. A block (br, bc) has rows 2*br..2*br+1, cols 2*bc..2*bc+1.
            # B block (br, bc) has rows 2*bc..2*bc+1, cols 2*bc..2*bc+1? No.
            # B is 4x4. Block (br, bc) of B has rows 2*bc..2*bc+1? No.
            # The block matrix of B is [[B00, B01], [B10, B11]].
            # B00 is rows 0,1 cols 0,1.
            # So block (br, bc) of B has rows 2*br..2*br+1, cols 2*bc..2*bc+1.
            # So the j indices (rows of B) are 2*br, 2*br+1.
            # The k indices (cols of B) are 2*bc, 2*bc+1.
            # So V4 indices for block (br, bc) are:
            # j*4+k where j in {2*br, 2*br+1}, k in {2*bc, 2*bc+1}.
            v_idx = [
                (2*br)*4 + 2*bc,
                (2*br)*4 + 2*bc + 1,
                (2*br+1)*4 + 2*bc,
                (2*br+1)*4 + 2*bc + 1
            ]
            
            # W4 indices: k*4+i.
            # C block (br, bc) has rows 2*br..2*br+1, cols 2*bc..2*bc+1.
            # So i in {2*br, 2*br+1}, k in {2*bc, 2*bc+1}.
            # W4 index is k*4+i.
            w_idx = [
                (2*bc)*4 + 2*br,
                (2*bc)*4 + 2*br + 1,
                (2*bc+1)*4 + 2*br,
                (2*bc+1)*4 + 2*br + 1
            ]
            
            # Now, for each Strassen multiplier m (0..6) and each inner Strassen multiplier s (0..6)
            # We add the factors.
            # The rank index is m*7 + s.
            for m in range(7):
                for s in range(7):
                    rank_idx = m * 7 + s
                    # U4 factor: U2[s][u_idx[m]]? No.
                    # U2[s] is the factor for the s-th multiplier of the inner Strassen.
                    # U2 is 4x7. U2[local_row][s].
                    # local_row corresponds to the element of the 2x2 block.
                    # For U4, the element is at u_idx[local_row].
                    # The value is U2[local_row][s] * U_block_factor[m]?
                    # Wait. The block Strassen has factors U2_block[local_row][m].
                    # The inner Strassen has factors U2_inner[local_row][s].
                    # So U4[u_idx[local_row]][rank_idx] += U2_block[local_row][m] * U2_inner[local_row][s]?
                    # No. The U matrix for the whole 4x4 is the tensor product of the block U and inner U?
                    # Actually, the construction is:
                    # M_block = sum_s (U_inner * A_block) (V_inner * B_block) W_inner^T
                    # A_block is a 2x2 matrix.
                    # This is correct.
                    
                    # We need U2_block. It is the same as U2, but applied to the block elements.
                    # U2_block[local_row][m] is the coefficient for the m-th block multiplier.
                    # U2_inner[local_row][s] is the coefficient for the s-th inner multiplier.
                    
                    # So for each local_row in 0..3:
                    for local_row in range(4):
                        coeff = U2[local_row][m] * U2[local_row][s]
                        if coeff != 0:
                            U4[u_idx[local_row]][rank_idx] += coeff
                            V4[v_idx[local_row]][rank_idx] += coeff
                            W4[w_idx[local_row]][rank_idx] += coeff
    
    # Now we have U4, V4, W4 for the 4x4x4 core.
    # We need to place them into the 5x5x5 U, V, W.
    # The core is i,j,k in 0..3.
    # U5 size 25xR. V5 size 25xR. W5 size 25xR.
    
    # Let's determine the total rank.
    # Core: 49.
    # Fringe: 61.
    # Total: 110.
    
    R = 110
    U5 = [[0]*R for _ in range(25)]
    V5 = [[0]*R for _ in range(25)]
    W5 = [[0]*R for _ in range(25)]
    
    # Place core factors (offset 0)
    # Core indices: i in 0..3, j in 0..3, k in 0..3.
    # U5[i*5+j] for i,j in 0..3.
    # V5[j*5+k] for j,k in 0..3.
    # W5[k*5+i] for k,i in 0..3.
    
    for i in range(4):
        for j in range(4):
            u_row_core = i * 4 + j
            u_row_5 = i * 5 + j
            for r in range(49):
                U5[u_row_5][r] = U4[u_row_core][r]
                
    for j in range(4):
        for k in range(4):
            v_row_core = j * 4 + k
            v_row_5 = j * 5 + k
            for r in range(49):
                V5[v_row_5][r] = V4[v_row_core][r]
                
    for k in range(4):
        for i in range(4):
            w_row_core = k * 4 + i
            w_row_5 = k * 5 + i
            for r in range(49):
                W5[w_row_5][r] = W4[w_row_core][r]
                
    # Now add fringe factors (offset 49)
    # Fringe blocks:
    # 1. i=4, j=4, k=4. Rank 1. Offset 49.
    U5[4*5+4][49] = 1
    V5[4*5+4][49] = 1
    W5[4*5+4][49] = 1
    
    # 2. i=4, j=4, k in 0..3. Rank 4. Offset 50.
    for k in range(4):
        col = 50 + k
        U5[4*5+4][col] = 1
        V5[4*5+k][col] = 1
        W5[k*5+4][col] = 1
        
    # 3. i=4, j in 0..3, k=4. Rank 4. Offset 54.
    for j in range(4):
        col = 54 + j
        U5[4*5+j][col] = 1
        V5[j*5+4][col] = 1
        W5[4*5+4][col] = 1
        
    # 4. i in 0..3, j=4, k=4. Rank 4. Offset 58.
    for i in range(4):
        col = 58 + i
        U5[i*5+4][col] = 1
        V5[4*5+4][col] = 1
        W5[4*5+i][col] = 1
        
    # 5. i=4, j in 0..3, k in 0..3. Rank 16. Offset 62.
    for j in range(4):
        for k in range(4):
            col = 62 + j*4 + k
            U5[4*5+j][col] = 1
            V5[j*5+k][col] = 1
            W5[k*5+4][col] = 1
            
    # 6. i in 0..3, j=4, k in 0..3. Rank 16. Offset 78.
    for i in range(4):
        for k in range(4):
            col = 78 + i*4 + k
            U5[i*5+4][col] = 1
            V5[4*5+k][col] = 1
            W5[k*5+i][col] = 1
            
    # 7. i in 0..3, j in 0..3, k=4. Rank 16. Offset 94.
    for i in range(4):
        for j in range(4):
            col = 94 + i*4 + j
            U5[i*5+j][col] = 1
            V5[j*5+4][col] = 1
            W5[4*5+i][col] = 1
            
    # Total rank used: 49 + 1 + 4 + 4 + 4 + 16 + 16 + 16 = 110.
    # Check if we exceeded R=110.
    # Max col: 94 + 15 = 109. So R=110 is correct.
    
    return {"U": U5, "V": V5, "W": W5}

# EVOLVE-BLOCK-END
