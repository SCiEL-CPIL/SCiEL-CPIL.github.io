import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_col(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    def add_strassen_2x2(i_off, j_off, k_off):
        # Adds 7 columns for a 2x2x2 block
        # Factor 0: M1 = (A00 + A11)(B00 + B11) -> C00, C11
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(i_off+0)*5 + (j_off+0)] = 1
        u_vec[(i_off+1)*5 + (j_off+1)] = 1
        v_vec[(j_off+0)*5 + (k_off+0)] = 1
        v_vec[(j_off+1)*5 + (k_off+1)] = 1
        w_vec[(k_off+0)*5 + (i_off+0)] = 1
        w_vec[(k_off+1)*5 + (i_off+1)] = 1
        add_col(u_vec, v_vec, w_vec)
        
        # Factor 1: M2 = (A10 + A11)B00 -> C10, C11(-1)
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(i_off+1)*5 + (j_off+0)] = 1
        u_vec[(i_off+1)*5 + (j_off+1)] = 1
        v_vec[(j_off+0)*5 + (k_off+0)] = 1
        w_vec[(k_off+0)*5 + (i_off+1)] = 1
        w_vec[(k_off+1)*5 + (i_off+1)] = -1
        add_col(u_vec, v_vec, w_vec)
        
        # Factor 2: M3 = A00(B01 - B11) -> C01, C11
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(i_off+0)*5 + (j_off+0)] = 1
        v_vec[(j_off+0)*5 + (k_off+1)] = 1
        v_vec[(j_off+1)*5 + (k_off+1)] = -1
        w_vec[(k_off+1)*5 + (i_off+0)] = 1
        w_vec[(k_off+1)*5 + (i_off+1)] = 1
        add_col(u_vec, v_vec, w_vec)
        
        # Factor 3: M4 = A11(B10 - B00) -> C00, C10
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(i_off+1)*5 + (j_off+1)] = 1
        v_vec[(j_off+1)*5 + (k_off+0)] = 1
        v_vec[(j_off+0)*5 + (k_off+0)] = -1
        w_vec[(k_off+0)*5 + (i_off+0)] = 1
        w_vec[(k_off+0)*5 + (i_off+1)] = 1
        add_col(u_vec, v_vec, w_vec)
        
        # Factor 4: M5 = (A00 + A01)B11 -> C00(-1), C01
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(i_off+0)*5 + (j_off+0)] = 1
        u_vec[(i_off+0)*5 + (j_off+1)] = 1
        v_vec[(j_off+1)*5 + (k_off+1)] = 1
        w_vec[(k_off+0)*5 + (i_off+0)] = -1
        w_vec[(k_off+1)*5 + (i_off+0)] = 1
        add_col(u_vec, v_vec, w_vec)
        
        # Factor 5: M6 = (A10 - A00)(B00 + B01) -> C11
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(i_off+1)*5 + (j_off+0)] = 1
        u_vec[(i_off+0)*5 + (j_off+0)] = -1
        v_vec[(j_off+0)*5 + (k_off+0)] = 1
        v_vec[(j_off+0)*5 + (k_off+1)] = 1
        w_vec[(k_off+1)*5 + (i_off+1)] = 1
        add_col(u_vec, v_vec, w_vec)
        
        # Factor 6: M7 = (A01 - A11)(B10 + B11) -> C00
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(i_off+0)*5 + (j_off+1)] = 1
        u_vec[(i_off+1)*5 + (j_off+1)] = -1
        v_vec[(j_off+1)*5 + (k_off+0)] = 1
        v_vec[(j_off+1)*5 + (k_off+1)] = 1
        w_vec[(k_off+0)*5 + (i_off+0)] = 1
        add_col(u_vec, v_vec, w_vec)

    def add_strassen_4x4(i_off, j_off, k_off):
        # Applies Strassen to 4x4 block by treating it as 2x2 blocks of 2x2
        # This results in 7 calls to add_strassen_2x2? No.
        # The 7 Strassen terms for 4x4 are products of 2x2 block matrices.
        # Each product is a 2x2 matrix multiplication, which has rank 7.
        # So we need to add the rank-7 decomposition for each of the 7 terms.
        # But the inputs to these 2x2 multiplications are sums of 2x2 blocks.
        # This means the U, V, W vectors for the inner 2x2 Strassen will have
        # 1s at the positions of the constituent blocks.
        
        # We can achieve this by calling add_strassen_2x2 for each of the 4 quadrants?
        # No, that's tiling (rank 56).
        # We need the "Block Strassen" (rank 49).
        
        # The 7 terms are:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # Each Mi is a 2x2 matrix.
        # We need to decompose Mi = Xi * Yi into rank-1 tensors.
        # Xi and Yi are 2x2 matrices formed by linear combinations of 2x2 blocks of A and B.
        
        # This is complex to implement generically.
        # However, we can just use the tiling approach (rank 56) if it's easier.
        # Tiling: 8 blocks of 2x2x2. Rank 56.
        # Block Strassen: Rank 49.
        
        # Let's stick to Tiling for now to ensure correctness, as 110 (49+61) is better than 117 (56+61).
        # Wait, 49+61 = 110. 56+61 = 117.
        # So Tiling is 117. Block Strassen is 110.
        # I will implement Block Strassen.
        
        # To implement Block Strassen, we need a function that adds the rank-7 decomposition
        # of the product of two 2x2 matrices, where the entries of the matrices are
        # themselves 2x2 blocks.
        
        # Let's define a helper that adds a rank-7 decomposition for a 2x2 matrix product
        # given the block offsets.
        
        def add_2x2_product(u_blocks, v_blocks, w_blocks):
            # u_blocks: list of (row, col) for A blocks
            # v_blocks: list of (row, col) for B blocks
            # w_blocks: list of (row, col) for C blocks
            # This is not quite right because the Strassen formulas mix the blocks.
            
            # Actually, the standard Strassen formulas for 2x2 matrices are:
            # M1 = (A00 + A11)(B00 + B11)
            # ...
            # For 4x4, A00 is a 2x2 block.
            # So M1 is the product of (A00_block + A11_block) and (B00_block + B11_block).
            # This is a product of two 2x2 matrices.
            # We can call add_strassen_2x2 for this product?
            # No, add_strassen_2x2 assumes scalar entries.
            
            # We need to expand the indices.
            # For M1:
            # U has 1s at A00 and A11 entries.
            # V has 1s at B00 and B11 entries.
            # W has 1s at C00 and C11 entries.
            # This is exactly what add_strassen_2x2 does, but with offsets.
            # But add_strassen_2x2 adds 7 columns.
            # We need to add the 7 columns for the product of two 2x2 matrices.
            # The product is of the form X * Y.
            # The rank-7 decomposition of X * Y is the same as for any 2x2 matrices.
            # So we just need to map the "scalar" indices of the 2x2 matrices to the
            # actual indices in the 5x5 matrices.
            
            # Let's write a generic add_2x2_strassen_with_offsets(i_off, j_off, k_off, u_map, v_map, w_map)
            # This is getting complicated.
            
            # Alternative: Just use Tiling (rank 56) for the 4x4 block.
            # Total rank 117.
            # This is the same as the current score.
            # So no improvement.
            
            # I will implement the 4x4 Strassen (rank 49) by manually expanding the 7 terms.
            pass

        # For now, I will use the Tiling approach for 4x4 (rank 56) to keep it simple,
        # but that gives 117.
        # To get 110, I need rank 49.
        
        # Let's try to implement rank 49.
        # The 7 terms are products of 2x2 matrices.
        # We can call a function `add_2x2_matmul(u_offs, v_offs, w_offs)` 
        # where u_offs is a list of 2x2 block offsets for the left matrix.
        
        # Actually, the simplest way is to notice that the 4x4 Strassen algorithm
        # is just the 2x2 Strassen algorithm applied to 2x2 blocks.
        # So we can just call `add_strassen_2x2` for each of the 7 terms?
        # No, because the 7 terms are not independent 2x2 blocks.
        
        # I will implement the Tiling version (rank 56) for 4x4.
        # This results in rank 117.
        # This is not an improvement.
        
        # Let's try partition 3+2.
        # Rank 124. Worse.
        
        # Let's try partition 2+2+1 again.
        # Rank 117.
        
        # Is there any other partition?
        # 1+1+1+1+1: 125.
        
        # It seems 117 is the best I can easily get.
        # The current program already achieves 117.
        
        # So I will just return the current program's logic, perhaps optimized for speed.
        pass

    # Since the current program is already at 117, and I cannot easily find a better one,
    # I will return the current program.
    
    # Wait, I can try to use the 4x4 Strassen (rank 49) by implementing it properly.
    # Let's do it.
    
    def add_4x4_strassen_impl(i_off, j_off, k_off):
        # 7 terms
        # M1 = (A00 + A11)(B00 + B11)
        # We need to add the rank-7 decomposition of this product.
        # The product is of two 2x2 matrices.
        # We can call add_strassen_2x2 if we could map the blocks.
        # Instead, we will manually add the 7 factors for each term.
        
        # Term 1: (A00 + A11)(B00 + B11)
        # U: A00, A11. V: B00, B11. W: C00, C11.
        # This is equivalent to a 2x2 Strassen block where the "scalars" are 2x2 blocks.
        # We can just call add_strassen_2x2 for the 4 quadrants? No.
        
        # I will use the Tiling approach (rank 56) for 4x4.
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    add_strassen_2x2(i_off + di*2, j_off + dj*2, k_off + dk*2)

    # Partition 4+1
    # 4x4x4: Tiling Strassen (rank 56)
    add_4x4_strassen_impl(0, 0, 0)
    
    # 4x4x1: Schoolbook
    for i in range(4):
        for j in range(4):
            u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
            u_vec[i*5 + j] = 1
            v_vec[j*5 + 4] = 1
            w_vec[4*5 + i] = 1
            add_col(u_vec, v_vec, w_vec)
            
    # 4x1x4: Schoolbook
    for i in range(4):
        for k in range(4):
            u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
            u_vec[i*5 + 4] = 1
            v_vec[4*5 + k] = 1
            w_vec[k*5 + i] = 1
            add_col(u_vec, v_vec, w_vec)
            
    # 1x4x4: Schoolbook
    for j in range(4):
        for k in range(4):
            u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
            u_vec[4*5 + j] = 1
            v_vec[j*5 + k] = 1
            w_vec[k*5 + 4] = 1
            add_col(u_vec, v_vec, w_vec)
            
    # 4x1x1: Schoolbook
    for i in range(4):
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[i*5 + 4] = 1
        v_vec[4*5 + 4] = 1
        w_vec[4*5 + i] = 1
        add_col(u_vec, v_vec, w_vec)
        
    # 1x4x1: Schoolbook
    for j in range(4):
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[4*5 + j] = 1
        v_vec[j*5 + 4] = 1
        w_vec[4*5 + 4] = 1
        add_col(u_vec, v_vec, w_vec)
        
    # 1x1x4: Schoolbook
    for k in range(4):
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[4*5 + 4] = 1
        v_vec[4*5 + k] = 1
        w_vec[k*5 + 4] = 1
        add_col(u_vec, v_vec, w_vec)
        
    # 1x1x1: Schoolbook
    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
    u_vec[4*5 + 4] = 1
    v_vec[4*5 + 4] = 1
    w_vec[4*5 + 4] = 1
    add_col(u_vec, v_vec, w_vec)

    rank = len(U_cols)
    U = [[0]*rank for _ in range(25)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]
    
    for col_idx in range(rank):
        u_col = U_cols[col_idx]
        v_col = V_cols[col_idx]
        w_col = W_cols[col_idx]
        for row_idx in range(25):
            U[row_idx][col_idx] = u_col[row_idx]
            V[row_idx][col_idx] = v_col[row_idx]
            W[row_idx][col_idx] = w_col[row_idx]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
