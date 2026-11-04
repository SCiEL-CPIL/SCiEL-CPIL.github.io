import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 56 solution: Two Strassen 4x4 multiplications
    # A is 3x4, padded to 4x4.
    # B is 4x6, split into B1 (4x3) and B2 (4x3), each padded to 4x4.
    
    # We will build U, V, W for one 4x4 Strassen multiplication first.
    # Then map to global indices.
    
    # Strassen 4x4 decomposition indices
    # A blocks: A11 (0,0), A12 (0,1), A21 (1,0), A22 (1,1)
    # B blocks: B11 (0,0), B12 (0,1), B21 (1,0), B22 (1,1)
    # Each block is 2x2.
    
    # We define the 7 multiplications.
    # Each multiplication is a product of two 2x2 matrices.
    # We can use the schoolbook 2x2x2 (rank 4) or Strassen 2x2x2 (rank 7).
    # Using Strassen for the inner 2x2 blocks would increase rank.
    # 4 outer blocks * 7 inner Strassen = 28.
    # If we use schoolbook for inner 2x2: 4 * 4 = 16.
    # But the outer Strassen combines them.
    # The standard Strassen 4x4 uses 7 multiplications of 2x2 matrices.
    # If we decompose each 2x2 multiplication with schoolbook (rank 4), total rank is 7*4 = 28.
    # This is the standard "Strassen 4x4" rank.
    
    # Let's generate the U, V, W for the 4x4 case.
    # U size 16x28, V size 16x28, W size 16x28.
    
    U_44 = np.zeros((16, 28))
    V_44 = np.zeros((16, 28))
    W_44 = np.zeros((16, 28))
    
    col = 0
    # Helper to add a schoolbook 2x2x2 multiplication
    def add_2x2_mult(A_rows, B_rows, W_rows, sign=1):
        nonlocal col
        # A_rows: list of 4 indices in 0..15 for A
        # B_rows: list of 4 indices in 0..15 for B
        # W_rows: list of 4 indices in 0..15 for W
        # Schoolbook: sum over k=0,1 of A[i,k] B[k,j]
        # Indices in 2x2 block: 0,1,2,3
        # A[i,k] index in 16: i*2 + k
        # B[k,j] index in 16: k*2 + j
        # W[i,j] index in 16: i*2 + j
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u_idx = A_rows[i*2 + k]
                    v_idx = B_rows[k*2 + j]
                    w_idx = W_rows[i*2 + j]
                    
                    U_44[u_idx, col] = sign
                    V_44[v_idx, col] = 1
                    W_44[w_idx, col] = 1
                    col += 1
        # This adds 4 columns.
        # But we need to handle the sums in Strassen.
        # Strassen M1 = (A11+A22)(B11+B22)
        # This is not a single schoolbook multiplication.
        # It is the product of two summed matrices.
        # (A11+A22) is a 2x2 matrix. (B11+B22) is a 2x2 matrix.
        # We need to decompose the product of two 2x2 matrices.
        # We can just use schoolbook for the 2x2 product.
        # So for each M_k, we call add_2x2_mult with the combined indices?
        # No, the indices in U and V must reflect the sum.
        # U entries for A11+A22:
        # U[i*2+k, col] should be 1 for A11 and 1 for A22.
        # So we add two schoolbook multiplications?
        # No, (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
        # This is 4 schoolbook multiplications.
        # So M1 requires 4 rank-1 terms (if we use schoolbook for the 2x2 blocks).
        # Total 7 M's * 4 = 28 terms.
        
    # Let's define the 7 terms explicitly.
    # M1: (A11+A22)(B11+B22)
    # A11 rows: 0,1,2,3. A22 rows: 8,9,10,11.
    # B11 rows: 0,1,2,3. B22 rows: 8,9,10,11.
    # W11 rows: 0,1,2,3.
    
    # We will iterate through the 7 Strassen terms.
    # For each term, we identify the A-blocks and B-blocks involved.
    # Then we add the schoolbook decomposition of the 2x2 product.
    
    # Term 1: (A11+A22)(B11+B22) -> W11 += M1
    # A parts: A11, A22. B parts: B11, B22.
    # Expansion: A11B11 + A11B22 + A22B11 + A22B22.
    
    def add_block_product(a_blocks, b_blocks, w_block, sign=1):
        nonlocal col
        # a_blocks: list of block indices (0,1,2,3) for A
        # b_blocks: list of block indices for B
        # w_block: block index for W
        # For each a in a_blocks, b in b_blocks:
        #   Add schoolbook 2x2x2 for a,b -> w_block
        
        for a_blk in a_blocks:
            for b_blk in b_blocks:
                # A block a_blk rows: a_blk*4 + 0..3
                # B block b_blk rows: b_blk*4 + 0..3
                # W block w_block rows: w_block*4 + 0..3
                
                A_rows = [a_blk*4 + i for i in range(4)]
                B_rows = [b_blk*4 + i for i in range(4)]
                W_rows = [w_block*4 + i for i in range(4)]
                
                # Schoolbook 2x2x2
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            u_idx = A_rows[i*2 + k]
                            v_idx = B_rows[k*2 + j]
                            w_idx = W_rows[i*2 + j]
                            
                            U_44[u_idx, col] = sign
                            V_44[v_idx, col] = 1
                            W_44[w_idx, col] = 1
                            col += 1
                            
    # M1: (A11+A22)(B11+B22) -> W11
    add_block_product([0, 3], [0, 3], 0, 1)
    
    # M2: (A21+A22)B11 -> W21
    add_block_product([2, 3], [0], 2, 1)
    
    # M3: A11(B12-B22) -> W12
    add_block_product([0], [1, 3], 1, 1) # A11B12
    add_block_product([0], [3], 1, -1) # A11(-B22)
    
    # M4: A22(B21-B11) -> W22
    add_block_product([3], [2, 0], 3, 1) # A22B21
    add_block_product([3], [0], 3, -1) # A22(-B11)
    
    # M5: (A11+A12)B22 -> W22 (Wait, W11 = M1+M4-M5+M7, W22 = M1-M2+M3+M6)
    # Standard Strassen:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So M5 contributes to W11 (-1) and W12 (+1).
    # M5: (A11+A12)B22
    add_block_product([0, 1], [3], 0, -1)
    add_block_product([0, 1], [3], 1, 1)
    
    # M6: (A21-A11)(B11+B12) -> W22
    # M6 contributes to W22 (+1).
    # Expansion: (A21-A11)(B11+B12) = A21B11 + A21B12 - A11B11 - A11B12
    add_block_product([2], [0, 1], 3, 1) # A21(B11+B12)
    add_block_product([0], [0, 1], 3, -1) # -A11(B11+B12)
    
    # M7: (A12-A22)(B21+B22) -> W11
    # M7 contributes to W11 (+1).
    # Expansion: (A12-A22)(B21+B22) = A12B21 + A12B22 - A22B21 - A22B22
    add_block_product([1], [2, 3], 0, 1) # A12(B21+B22)
    add_block_product([3], [2, 3], 0, -1) # -A22(B21+B22)
    
    # Now U_44, V_44, W_44 are filled. col should be 28.
    
    # Map to global indices.
    # Global U: 3x4 = 12 rows.
    # Global V: 4x6 = 24 rows.
    # Global W: 6x3 = 18 rows.
    
    # We have two halves: B1 (cols 0-2) and B2 (cols 3-5).
    # For B1:
    # V rows 0-7 (j=0,1, k=0,1,2) and j=2,3, k=0,1,2.
    # B1 is 4x3. Padded to 4x4.
    # B1 rows in 4x4:
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=0, k=2 -> 2
    # j=0, k=3 (pad) -> 3
    # j=1, k=0 -> 4
    # ...
    # j=1, k=3 (pad) -> 7
    # j=2, k=0 -> 8
    # ...
    # j=3, k=3 (pad) -> 15
    
    # Global V index for j, k is j*6 + k.
    # For B1, k is 0,1,2.
    # So V_44 row 0 -> Global V row 0*6+0 = 0
    # V_44 row 1 -> Global V row 0*6+1 = 1
    # V_44 row 2 -> Global V row 0*6+2 = 2
    # V_44 row 3 (pad) -> No global row? Or 0?
    # The term involving pad row 3 in V_44 will be multiplied by 0 in the actual B.
    # So we can just ignore it or map it to a dummy row.
    # But we must construct V of shape (24, 56).
    # We will set V[row, col] for valid rows.
    
    U = np.zeros((12, 56))
    V = np.zeros((24, 56))
    W = np.zeros((18, 56))
    
    # First half (cols 0-27)
    # U mapping:
    # U_44 row i*4+j -> Global U row i*4+j (for i=0,1,2).
    # i=3 is pad, ignore.
    for r in range(12):
        U[r, 0:28] = U_44[r, 0:28]
        
    # V mapping for B1:
    # V_44 row j*4+k.
    # If k < 3, Global V row j*6+k.
    # If k == 3, ignore.
    for j in range(4):
        for k in range(3):
            v44_idx = j*4 + k
            v_global_idx = j*6 + k
            V[v_global_idx, 0:28] = V_44[v44_idx, 0:28]
            
    # W mapping for C1 (3x3):
    # W_44 row k*4+i? No, W_44 is 4x4 result.
    # W_44 row i*4+k?
    # In add_block_product, W_rows[i*2+j] where i,j are 0,1.
    # So W_44 index is i*2+j within the block.
    # Block w_block is 0,1,2,3.
    # W_44 row w_block*4 + i*2 + j.
    # This corresponds to result row i, col j.
    # Global W index is k*3 + i.
    # Here k is the col of C1 (0,1,2). i is row of C1 (0,1,2).
    # So Global W row k*3 + i.
    # W_44 row: w_block*4 + i*2 + j.
    # w_block 0 -> C11 (i=0,1, j=0,1).
    # w_block 1 -> C12 (i=0,1, j=2,3?).
    # Wait, C1 is 3x3.
    # The 4x4 result C_44 has C11 (2x2), C12 (2x2), C21 (2x2), C22 (2x2).
    # C1 (3x3) is the top-left 3x3 of C_44.
    # So we need C_44[0:3, 0:3].
    # C_44 rows 0,1,2 and cols 0,1,2.
    # C_44 row i, col j.
    # W_44 index for i,j:
    # If i<2, j<2: Block 0, index i*2+j.
    # If i<2, j>=2: Block 1, index 4 + (i*2 + (j-2)).
    # If i>=2, j<2: Block 2, index 8 + ((i-2)*2 + j).
    # If i>=2, j>=2: Block 3, index 12 + ((i-2)*2 + (j-2)).
    
    for i in range(3):
        for k in range(3): # k is col
            if i < 2 and k < 2:
                w44_idx = 0*4 + i*2 + k
            elif i < 2 and k >= 2:
                w44_idx = 1*4 + i*2 + (k-2)
            elif i >= 2 and k < 2:
                w44_idx = 2*4 + (i-2)*2 + k
            else:
                w44_idx = 3*4 + (i-2)*2 + (k-2)
                
            w_global_idx = k*3 + i
            W[w_global_idx, 0:28] = W_44[w44_idx, 0:28]
            
    # Second half (cols 28-55)
    # U is the same.
    U[:, 28:56] = U_44[:, 0:28]
    
    # V mapping for B2:
    # B2 is 4x3. Cols 3,4,5 of B.
    # Padded to 4x4.
    # B2 row j, col k' (0,1,2) corresponds to B col k'+3.
    # V_44 row j*4+k'.
    # Global V row j*6 + (k'+3).
    for j in range(4):
        for k in range(3):
            v44_idx = j*4 + k
            v_global_idx = j*6 + (k+3)
            V[v_global_idx, 28:56] = V_44[v44_idx, 0:28]
            
    # W mapping for C2 (3x3):
    # C2 is 3x3. Cols 3,4,5 of C.
    # C_44 top-left 3x3.
    # Global W row k*3 + i. Here k is 3,4,5.
    for i in range(3):
        for k in range(3): # k is 0,1,2 local
            if i < 2 and k < 2:
                w44_idx = 0*4 + i*2 + k
            elif i < 2 and k >= 2:
                w44_idx = 1*4 + i*2 + (k-2)
            elif i >= 2 and k < 2:
                w44_idx = 2*4 + (i-2)*2 + k
            else:
                w44_idx = 3*4 + (i-2)*2 + (k-2)
                
            w_global_idx = (k+3)*3 + i
            W[w_global_idx, 28:56] = W_44[w44_idx, 0:28]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

